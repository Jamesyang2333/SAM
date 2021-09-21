# SAM
### Instruction
We have provided pretrained model for both Census and DMV dataset. To generate database from pretrained models using SAM, use the following commands.
```
python gen_data_model.py --dataset census --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking --glob census-pretrained.pt --save-name census
python gen_data_model.py --dataset dmv --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking --glob dmv-pretrained.pt --save-name dmv
```
The generated relation is saved at `./generated_data_tables`.
