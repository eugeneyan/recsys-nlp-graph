import argparse
import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader

from src.config import MODEL_PATH
from src.ml.data_loader_edges import Edges, EdgesDataset
from src.ml.mf_continuous import MFContinuous
from src.utils.logger import logger

shuffle = True
emb_dim = 128
epochs = 5
initial_lr = 0.01

# Torch parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Device: {}, emb_dim: {}, epochs: {}, initial_lr: {}'.format(device, emb_dim, epochs, initial_lr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training embeddings on torch')
    parser.add_argument('read_path', type=str, help='Path to sequences.npy')
    parser.add_argument('val_path', type=str, help='Path to val.csv')
    parser.add_argument('val_samp_path', type=str, help='Path to val_samp.csv')
    parser.add_argument('batch_size', type=int, help='Batchsize for dataloader')
    parser.add_argument('n_workers', type=int, help='Number of workers for dataloader')
    args = parser.parse_args()

    # Initialize dataset
    edges = Edges(args.read_path, args.val_path)
    dataset = EdgesDataset(edges)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.n_workers,
                            collate_fn=dataset.collate_continuous)

    # Initialize validation set
    val_samp = pd.read_csv(args.val_samp_path)

    # Get product ID
    word2id_func = np.vectorize(edges.get_product_id)
    val_samp['product1_id'] = word2id_func(val_samp['product1'].values)
    val_samp['product2_id'] = word2id_func(val_samp['product2'].values)
    val_samp = val_samp[(val_samp['product1_id'] > -1) & (val_samp['product2_id'] > -1)]  # Keep those with valid ID
    logger.info('No. of validation samples: {}'.format(val_samp.shape[0]))

    product1_id = val_samp['product1_id'].values
    product2_id = val_samp['product2_id'].values

    # Initialize model
    mf = MFContinuous(edges.n_unique_tokens, emb_dim).to(device)

    # Train loop
    optimizer = optim.Adam(mf.parameters(), lr=initial_lr)

    results = []
    start_time = datetime.datetime.now()
    for epoch in range(epochs):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))
        running_loss = 0

        # Training loop
        for i, batches in enumerate(dataloader):

            product1 = batches[0].to(device)
            product2 = batches[1].to(device)
            label = batches[2].to(device)

            optimizer.zero_grad()

            pred = mf.forward(product1, product2)
            loss = mf.loss(pred, label)
            loss.backward()
            optimizer.step()

            scheduler.step()
            running_loss = running_loss * 0.9 + loss.item() * 0.1

            if i > 0 and i % 1000 == 0:
                # Validation Check
                with torch.no_grad():
                    pred = mf.predict(torch.LongTensor(val_samp['product1_id']).to(device),
                                      torch.LongTensor(val_samp['product2_id']).to(device))
                    score = roc_auc_score(val_samp['edge'], pred.detach().cpu().numpy())

                logger.info("Epoch: {}, Seq: {:,}/{:,}, " \
                            "Loss: {:.4f}, AUC-ROC: {:.4f}, Lr: {:.6f}".format(epoch, i, len(dataloader), running_loss,
                                                                               score, optimizer.param_groups[0]['lr']))
                results.append([epoch, i, running_loss, score])
                running_loss = 0

        # save model
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
        state_dict_path = '{}/mf_continuous_edges_epoch_{}_{}.pt'.format(MODEL_PATH, epoch, current_datetime)
        torch.save(mf.state_dict(), state_dict_path)
        logger.info('Model state dict saved to {}'.format(state_dict_path))

    end_time = datetime.datetime.now()
    time_diff = round((end_time - start_time).total_seconds() / 60, 2)
    logger.info('Total time taken: {:,} minutes'.format(time_diff))

    # Save results
    results_df = pd.DataFrame(results, columns=['epoch', 'batches', 'loss', 'auc'])
    results_df.to_csv('{}/model_metrics_mf_continuous_edges.csv'.format(MODEL_PATH), index=False)
