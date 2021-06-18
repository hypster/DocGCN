### DocGCN

Note:

- When first run some script, there will be hints about file not exist. Simply create the required folders will solve the issue.


Example how to use:

* common step:

  * `python prepareCorpus.py --file 20ng`

  * `python prepareTokenizedText.py --file 20ng --max_df 1.0 --min_df 5`

* To train DocGCN:
 
    * `python buildDocumentGraph.py --file 20ng  --window 5 --cutoff 0`
    
    * `python GCNP.py --file 20ng --num_layers 2 --hidden_dim 64 --dropout 0.5 --lr 0.001 --epochs 20 --batch_size 64 --train_val_ratio 0.9`

* Optionally, to train TextGCN:

    * `python buildCorpusGraph.py --file 20ng --window 5 --max_df 1.0 --min_df 5 --cutoff 0`
    
    * `python GCN.py --file 20ng --num_layers 2 --hidden_dim 64 --dropout 0.5 --lr 0.02 --epochs 1000`


