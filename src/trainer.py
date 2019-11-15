import gc

import numpy as np
import torch

from logger import LOGGER


def train_one_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=1,
                    steps_upd_logging=1000, any_drop=False):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()

        logits = model(features)
        if any_drop:
            loss = criterion(logits, targets[:, 1:])
            loss += criterion(torch.max(logits, axis=1)[0], targets[:, 0]) * 2 / 5
        else:
            loss = criterion(logits, targets)

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

    return total_loss / (step + 1)


def predict(model, test_loader, device, n_tta=1, flip_aug=False):
    model.eval()

    preds_cat = []

    with torch.no_grad():
        for step, imgs in enumerate(test_loader):
            features = imgs[0].to(device)

            if flip_aug:
                logits = model(features)[:, :6]
                if n_tta >= 2:
                    flip_img = imgs[1].to(device)
                    logits += model(flip_img)[:, 6:]
                    del flip_img
                    gc.collect()

                if n_tta >= 4:
                    img_tta = imgs[2].to(device)
                    logits += model(img_tta)[:, :6]
                    del img_tta
                    gc.collect()

                    img_tta_flip = imgs[3].to(device)
                    logits += model(img_tta_flip)[:, 6:]
                    del img_tta_flip
                    gc.collect()
            else:
                logits = model(features)
                if n_tta >= 2:
                    flip_img = imgs[1].to(device)
                    logits += model(flip_img)
                    del flip_img
                    gc.collect()

                if n_tta >= 4:
                    img_tta = imgs[2].to(device)
                    logits += model(img_tta)
                    del img_tta
                    gc.collect()

                    img_tta_flip = imgs[3].to(device)
                    logits += model(img_tta_flip)
                    del img_tta_flip
                    gc.collect()

            logits = logits / n_tta

            del imgs
            gc.collect()

            logits = torch.sigmoid(logits).float().cpu().numpy()
            preds_cat.append(logits)

        all_preds = np.concatenate(preds_cat, axis=0)


    return all_preds


def predict_external(model, test_loader, device, n_tta=1, flip_aug=False):
    model.eval()

    preds_cat = []
    is_dicoms = []

    with torch.no_grad():
        for step, (imgs, is_dicom) in enumerate(test_loader):
            features = imgs[0].to(device)

            if flip_aug:
                logits = model(features)[:, :6]
                if n_tta >= 2:
                    flip_img = imgs[1].to(device)
                    logits += model(flip_img)[:, 6:]
                    del flip_img
                    gc.collect()

                if n_tta >= 4:
                    img_tta = imgs[2].to(device)
                    logits += model(img_tta)[:, :6]
                    del img_tta
                    gc.collect()

                    img_tta_flip = imgs[3].to(device)
                    logits += model(img_tta_flip)[:, 6:]
                    del img_tta_flip
                    gc.collect()
            else:
                logits = model(features)
                if n_tta >= 2:
                    flip_img = imgs[1].to(device)
                    logits += model(flip_img)
                    del flip_img
                    gc.collect()

                if n_tta >= 4:
                    img_tta = imgs[2].to(device)
                    logits += model(img_tta)
                    del img_tta
                    gc.collect()

                    img_tta_flip = imgs[3].to(device)
                    logits += model(img_tta_flip)
                    del img_tta_flip
                    gc.collect()

            logits = logits / n_tta

            del imgs
            gc.collect()

            logits = torch.sigmoid(logits).float().cpu().numpy()
            preds_cat.append(logits)
            is_dicoms.extend(list(is_dicom.numpy()))

            #if step % 1:
            LOGGER.info(np.sum(is_dicoms))


        all_preds = np.concatenate(preds_cat, axis=0)


    return all_preds, np.array(is_dicoms)
