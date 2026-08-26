import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score

import config
from aspnet.losses import MaskedCELoss, MaskedMSELoss
from aspnet.utils import Logger, build_model, create_loaders, generate_inputs, generate_mask, seed_everything


REGRESSION = "regression"
CLASSIFICATION = "classification"


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_selected_expert_state(model, source_state, expert_name):
    current_state = model.state_dict()
    suffix = f"_{expert_name}"
    projection_prefix = {
        "a": "a_in_proj",
        "t": "t_in_proj",
        "v": "v_in_proj",
    }[expert_name]
    for key, value in source_state.items():
        is_expert_parameter = suffix in key or key.startswith(projection_prefix)
        if is_expert_parameter and key in current_state:
            current_state[key] = value.to(current_state[key].device)
    model.load_state_dict(current_state)


def run_epoch(args, model, reg_loss, cls_loss, dataloader, optimizer=None, train=False, first_stage=True):
    if train:
        model.train()
    else:
        model.eval()

    preds, preds_a, preds_t, preds_v, labels, masks, losses = [], [], [], [], [], [], []
    task = task_type(args.dataset)
    cuda = torch.cuda.is_available() and not args.no_cuda

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8]
        prompt_features = data[9] if (args.use_semantic_alignment and len(data) > 10) else None
        prompt_features_for_model = None if first_stage else prompt_features

        seq_len, batch_size = audio_host.size(0), audio_host.size(1)
        host_mask = _make_modality_masks(seq_len, batch_size, args.test_condition, first_stage)
        guest_mask = _make_modality_masks(seq_len, batch_size, args.test_condition, first_stage)

        audio_host_mask, text_host_mask, visual_host_mask = host_mask
        audio_guest_mask, text_guest_mask, visual_guest_mask = guest_mask
        masked_audio_host = audio_host * audio_host_mask
        masked_text_host = text_host * text_host_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        if cuda:
            masked_audio_host, audio_host_mask = masked_audio_host.to(args.device), audio_host_mask.to(args.device)
            masked_text_host, text_host_mask = masked_text_host.to(args.device), text_host_mask.to(args.device)
            masked_visual_host, visual_host_mask = masked_visual_host.to(args.device), visual_host_mask.to(args.device)
            masked_audio_guest, audio_guest_mask = masked_audio_guest.to(args.device), audio_guest_mask.to(args.device)
            masked_text_guest, text_guest_mask = masked_text_guest.to(args.device), text_guest_mask.to(args.device)
            masked_visual_guest, visual_guest_mask = masked_visual_guest.to(args.device), visual_guest_mask.to(args.device)
            qmask, umask, label = qmask.to(args.device), umask.to(args.device), label.to(args.device)
            if prompt_features is not None:
                prompt_features = prompt_features.to(args.device)
                prompt_features_for_model = None if first_stage else prompt_features

        input_features = generate_inputs(
            masked_audio_host,
            masked_text_host,
            masked_visual_host,
            masked_audio_guest,
            masked_text_guest,
            masked_visual_guest,
            qmask,
        )
        input_masks = generate_inputs(
            audio_host_mask,
            text_host_mask,
            visual_host_mask,
            audio_guest_mask,
            text_guest_mask,
            visual_guest_mask,
            qmask,
        )

        _hidden, out, out_a, out_t, out_v, _weights = model(
            input_features[0],
            input_masks[0],
            umask,
            first_stage,
            prompt_features=prompt_features_for_model,
        )

        lp = out.view(-1, out.size(2))
        lp_a = out_a.view(-1, out_a.size(2))
        lp_t = out_t.view(-1, out_t.size(2))
        lp_v = out_v.view(-1, out_v.size(2))
        labels_flat = label.view(-1)

        if task == CLASSIFICATION:
            if first_stage:
                loss_a = cls_loss(lp_a, labels_flat, umask)
                loss_t = cls_loss(lp_t, labels_flat, umask)
                loss_v = cls_loss(lp_v, labels_flat, umask)
                batch_loss = (loss_a + loss_t + loss_v) / 3
            else:
                batch_loss = cls_loss(lp, labels_flat, umask)
        else:
            if first_stage:
                loss_a = reg_loss(lp_a, labels_flat, umask)
                loss_t = reg_loss(lp_t, labels_flat, umask)
                loss_v = reg_loss(lp_v, labels_flat, umask)
                batch_loss = (loss_a + loss_t + loss_v) / 3
            else:
                batch_loss = reg_loss(lp, labels_flat, umask)

        if train:
            batch_loss.backward()
            optimizer.step()

        losses.append(batch_loss.item() * torch.sum(umask).item())
        preds.append(lp.detach().cpu().numpy())
        preds_a.append(lp_a.detach().cpu().numpy())
        preds_t.append(lp_t.detach().cpu().numpy())
        preds_v.append(lp_v.detach().cpu().numpy())
        labels.append(labels_flat.detach().cpu().numpy())
        masks.append(umask.view(-1).detach().cpu().numpy())

    return compute_metrics(args.dataset, preds, preds_a, preds_t, preds_v, labels, masks, losses)


def _make_modality_masks(seq_len, batch_size, condition, first_stage):
    matrix = generate_mask(seq_len, batch_size, condition, first_stage)
    masks = []
    for values in matrix:
        mask = np.reshape(values, (batch_size, seq_len, 1))
        masks.append(torch.LongTensor(mask.transpose(1, 0, 2)))
    return masks


def compute_metrics(dataset, preds, preds_a, preds_t, preds_v, labels, masks, losses):
    preds = np.concatenate(preds)
    preds_a = np.concatenate(preds_a)
    preds_t = np.concatenate(preds_t)
    preds_v = np.concatenate(preds_v)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)
    avg_loss = round(np.sum(losses) / np.sum(masks), 4)

    if task_type(dataset) == CLASSIFICATION:
        pred_labels = np.argmax(preds, 1)
        pred_a = np.argmax(preds_a, 1)
        pred_t = np.argmax(preds_t, 1)
        pred_v = np.argmax(preds_v, 1)
        return {
            "mae": 0.0,
            "corr": recall_score(labels, pred_labels, sample_weight=masks, average="macro"),
            "acc": accuracy_score(labels, pred_labels, sample_weight=masks),
            "f1": f1_score(labels, pred_labels, sample_weight=masks, average="weighted"),
            "experts": [
                accuracy_score(labels, pred_a, sample_weight=masks),
                accuracy_score(labels, pred_t, sample_weight=masks),
                accuracy_score(labels, pred_v, sample_weight=masks),
            ],
            "loss": avg_loss,
        }

    non_zero = np.array([idx for idx, value in enumerate(labels) if value != 0])
    pred_values = preds.squeeze()
    return {
        "mae": np.mean(np.absolute(labels[non_zero] - pred_values[non_zero])),
        "corr": np.corrcoef(labels[non_zero], pred_values[non_zero])[0][1],
        "acc": accuracy_score((labels[non_zero] > 0), (pred_values[non_zero] > 0)),
        "f1": f1_score((labels[non_zero] > 0), (pred_values[non_zero] > 0), average="weighted"),
        "experts": [
            accuracy_score((labels[non_zero] > 0), (preds_a[non_zero] > 0)),
            accuracy_score((labels[non_zero] > 0), (preds_t[non_zero] > 0)),
            accuracy_score((labels[non_zero] > 0), (preds_v[non_zero] > 0)),
        ],
        "loss": avg_loss,
    }


def task_type(dataset):
    return CLASSIFICATION if dataset in config.CLASSIFICATION_DATASETS else REGRESSION


def metric_for_selection(dataset, metrics):
    return metrics["acc"] if task_type(dataset) == CLASSIFICATION else metrics["f1"]


def configure_dataset(args):
    args.dataset = config.normalize_dataset_name(args.dataset, args.iemocap_classes)
    if args.dataset in config.REGRESSION_DATASETS:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    elif args.dataset == "IEMOCAP4":
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
        args.use_semantic_alignment = False
    elif args.dataset == "IEMOCAP6":
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2
        args.use_semantic_alignment = False
    return args


def resolve_prompt_root(args):
    if not args.use_semantic_alignment:
        return None
    if args.prompt_feature == "auto":
        candidates = [
            f"prompt-deberta-large-4-UTT-{args.test_condition}",
            "prompt-deberta-large-4-UTT",
        ]
        for candidate in candidates:
            path = os.path.join(config.PATH_TO_FEATURES[args.dataset], candidate)
            if os.path.exists(path):
                args.prompt_feature = candidate
                break
    path = os.path.join(config.PATH_TO_FEATURES[args.dataset], args.prompt_feature)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt feature directory does not exist: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Train ASPNet for incomplete multimodal learning.")
    parser.add_argument("--dataset", default="CMU-MOSI", help="CMU-MOSI, CMU-MOSEI, or IEMOCAP.")
    parser.add_argument("--iemocap-classes", type=int, choices=[4, 6], default=4)
    parser.add_argument("--audio-feature", default="wav2vec-large-c-UTT")
    parser.add_argument("--text-feature", default="deberta-large-4-UTT")
    parser.add_argument("--video-feature", default="manet_UTT")
    parser.add_argument("--prompt-feature", default="auto")
    parser.add_argument("--test_condition", default="atv", choices=["a", "t", "v", "at", "av", "tv", "atv"])
    parser.add_argument("--disable-semantic-alignment", dest="use_semantic_alignment", action="store_false")
    parser.set_defaults(use_semantic_alignment=True)

    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--drop_rate", type=float, default=0.5)
    parser.add_argument("--attn_drop_rate", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=256)

    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--stage_epoch", type=int, default=150)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def main():
    args = configure_dataset(parse_args())
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    seed_everything(args.seed)

    run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{args.dataset}_{args.test_condition}"
    log_dir = os.path.join(config.LOG_DIR, args.dataset)
    sys.stdout = Logger(os.path.join(log_dir, f"{run_name}.log"), stream=sys.stdout)
    sys.stderr = sys.stdout
    print(args)

    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], args.audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], args.text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], args.video_feature)
    prompt_root = resolve_prompt_root(args)
    for path in (audio_root, text_root, video_root):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature directory does not exist: {path}")

    loader_outputs = create_loaders(
        args.dataset,
        audio_root,
        text_root,
        video_root,
        args.batch_size,
        args.num_workers,
        prompt_root=prompt_root,
    )
    train_loaders, valid_loaders, test_loaders, feat_dims = loader_outputs
    adim, tdim, vdim = feat_dims[:3]
    pdim = feat_dims[3] if len(feat_dims) == 4 else 0

    fold_results, fold_models = [], []
    for fold_idx, (train_loader, valid_loader, test_loader) in enumerate(zip(train_loaders, valid_loaders, test_loaders)):
        print(f">>>>> Fold {fold_idx + 1}/{len(train_loaders)} >>>>>")
        model = build_model(args, adim, tdim, vdim, pdim=pdim).to(args.device)
        reg_loss, cls_loss = MaskedMSELoss(), MaskedCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        valid_expert_scores = {"a": [], "t": [], "v": []}
        valid_scores, states = [], []
        start = time.time()

        for epoch in range(args.epochs):
            first_stage = epoch < args.stage_epoch
            train_metrics = run_epoch(args, model, reg_loss, cls_loss, train_loader, optimizer, True, first_stage)
            valid_metrics = run_epoch(args, model, reg_loss, cls_loss, valid_loader, None, False, first_stage)

            states.append(clone_state_dict(model))
            valid_scores.append(-np.inf if first_stage else metric_for_selection(args.dataset, valid_metrics))
            for idx, name in enumerate(("a", "t", "v")):
                valid_expert_scores[name].append(valid_metrics["experts"][idx])

            if first_stage:
                print(
                    f"epoch:{epoch}; "
                    f"a_train:{train_metrics['experts'][0]:.3f}; t_train:{train_metrics['experts'][1]:.3f}; v_train:{train_metrics['experts'][2]:.3f}; "
                    f"a_valid:{valid_metrics['experts'][0]:.3f}; t_valid:{valid_metrics['experts'][1]:.3f}; v_valid:{valid_metrics['experts'][2]:.3f}"
                )
            else:
                print(
                    f"epoch:{epoch}; "
                    f"train_acc:{train_metrics['acc']:.4f}; train_f1:{train_metrics['f1']:.4f}; train_loss:{train_metrics['loss']}; "
                    f"valid_acc:{valid_metrics['acc']:.4f}; valid_f1:{valid_metrics['f1']:.4f}; valid_loss:{valid_metrics['loss']}"
                )

            if epoch == args.stage_epoch - 1:
                for expert in ("a", "t", "v"):
                    best_idx = int(torch.argmax(torch.Tensor(valid_expert_scores[expert])))
                    load_selected_expert_state(model, states[best_idx], expert)
                    print(f"selected_{expert}_expert_epoch: {best_idx}")
                states[-1] = clone_state_dict(model)

        best_epoch = int(np.argmax(np.array(valid_scores)))
        model.load_state_dict(states[best_epoch])
        test_metrics = run_epoch(args, model, reg_loss, cls_loss, test_loader, None, False, False)
        fold_results.append(test_metrics)
        fold_models.append(clone_state_dict(model))
        print(f"fold:{fold_idx}; best_valid_epoch:{best_epoch}; test:{test_metrics}; duration:{time.time() - start:.1f}s")

    save_dir = args.save_dir or os.path.join(config.MODEL_DIR, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    mean = {key: float(np.mean([fold[key] for fold in fold_results])) for key in ("mae", "corr", "acc", "f1")}
    save_name = (
        f"{run_name}_hidden-{args.hidden}_bs-{args.batch_size}_"
        f"mae-{mean['mae']:.3f}_corr-{mean['corr']:.3f}_f1-{mean['f1']:.4f}_acc-{mean['acc']:.4f}.pth"
    )
    save_path = os.path.join(save_dir, save_name)
    torch.save({"model": fold_models[0] if len(fold_models) == 1 else fold_models, "args": vars(args), "metrics": mean}, save_path)
    print(f"mean_test_metrics: {mean}")
    print(f"saved_model: {save_path}")


if __name__ == "__main__":
    main()
