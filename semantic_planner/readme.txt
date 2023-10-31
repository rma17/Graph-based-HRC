This is the training codes for semantic planner
Prerequisite:
torch_geometric

Pipeline:
!!! Note the file root should be changed !!!
run HRC_Encoder1.py for training GNN and save nodes embeddings
run HRC_Decoder.py for training LSTM for planning result
run test.py  for testing the overall performance