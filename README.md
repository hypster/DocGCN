# Work for the thesis
Example how to use:

#### prepare the corpus 
`python prepare.py --file 20ng`

#### To build corpus level graph:

run with default setting: `python buildCorpusGraph.py --file 20ng`

run custom setting: `python buildCorpusGraph.py --file 20ng --window 5 --max_df 0.8 --min_df 5 --cutoff 0`

#### To build document level graph 

`python buildDocumentGraph.py --file 20ng`

run with custom setting: `python buildDocumentGraph.py --file 20ng  --window 5 --cutoff 0`

#### train TextGCN: 

run with default setting: `python GCN.py --file 20ng`

run custom setting: `python GCN.py --file 20ng --num_layers 2 --hidden_dim 64 --dropout 0.5 --lr 0.02 --epochs 1000`

#### train GCN-P

run with default setting: `python GCNP.py --file 20ng`

run with custom setting: `python GCNP.py --file 20ng --num_layers 2 --hidden_dim 32 --dropout 0.5 --lr 0.001 --epochs 10 --batch_size 1024 --train_val_ratio 0.8`

