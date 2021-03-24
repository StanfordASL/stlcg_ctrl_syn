Code accompanying Adaptive Trajectory-feedback Control with Signal Temporal Logic Specifications

More documentation to come.

See `ctrl_syn/src/learning.py` see the neural network architecture.

To train a model, run
`python3 run_cnn.py --iter_max 200 --lstm_dim 256 --device cuda --dropout 0.2 --weight_ctrl 0.5 --weight_recon 0.7 --weight_stl -1 --teacher_training -1 --mode adv_training_iteration_rapid --type drive --stl_scale -1 --status new --stl_max 0.5 --stl_min 0.1 --scale_min 0.1 --scale_max 50.0 --trainset_size 128 --evalset_size 32 --number 0 --action create --adv_iter_max 100 --run 0 --expert_mini_bs 1`

