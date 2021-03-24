Code accompanying "Adaptive Trajectory-feedback Control with Signal Temporal Logic Specifications" by Karen Leung and Marco Pavone

More documentation to come.




## Neural network architecture parameters}
The size of our neural network is described below. Note that the image <img src="https://render.githubusercontent.com/render/math?math=e"> is of size (number of channels) <img src="https://render.githubusercontent.com/render/math?math=\times 480 \times 480">. In our experiments, there were four channels. The state dimension of our system is <img src="https://render.githubusercontent.com/render/math?math=n=4"> and with control input of size <img src="https://render.githubusercontent.com/render/math?math=m=2">.
- Summarizing <img src="https://render.githubusercontent.com/render/math?math=e"> into a hidden state (<img src="https://render.githubusercontent.com/render/math?math=g_\mathrm{CNN}^{|e|\rightarrow n_c}">): Our CNN has 2 layers. The first convolution layer applies four 8x8 filters with a stride of 4 and 4 padded zeros per spatial dimension. This is followed by batch normalization, the ReLU nonlinear activation function, and 4x4 max-pooling. The second convolution layer then applies a single 8x8 filter with a stride of 4 and 4 padded zeros per spatial dimension to yield the the final CNN output <img src="https://render.githubusercontent.com/render/math?math=c_\mathcal{E}">. This results in <img src="https://render.githubusercontent.com/render/math?math=n_c = 64">.
- Initializing LSTM hidden state (<img src="https://render.githubusercontent.com/render/math?math=g_\mathrm{MLP}^{n_c \rightarrow n_h}">):  Two identically-sized three layer MLPs are used, each to project <img src="https://render.githubusercontent.com/render/math?math=c_\mathcal{E}"> into each component of the LSTM hidden state which is a 2-tuple of a <img src="https://render.githubusercontent.com/render/math?math=n_h"> dimensional vector. Each MLP if of the form: a linear layer  of size <img src="https://render.githubusercontent.com/render/math?math=(64 + n, n_h)"> with bias, followed by a <img src="https://render.githubusercontent.com/render/math?math=\mathrm{Tanh}()"> activation, a linear layer <img src="https://render.githubusercontent.com/render/math?math=(n_h, n_h)"> with bias, another <img src="https://render.githubusercontent.com/render/math?math=\mathrm{Tanh}()"> activation, and then another linear layer <img src="https://render.githubusercontent.com/render/math?math=(n_h. n_h)">.
- LSTM cell (<img src="https://render.githubusercontent.com/render/math?math=g_\mathrm{LSTM}^{n\rightarrow n_h}">): An LSTM cell with hidden state size of <img src="https://render.githubusercontent.com/render/math?math=n_h = 128)">.
- Projecting hidden state to control (<img src="https://render.githubusercontent.com/render/math?math=g_\mathrm{MLP}^{n_h \rightarrow m}">): A single linear layer of size <img src="https://render.githubusercontent.com/render/math?math=(n_h, m)"> with bias.

## Training hyperparameters
We detail the hyperparameters used in your offline training process below.
- We use the Adam optimizer with PyTorch's default parameters and a weight decay = 0.05.
- <img src="https://render.githubusercontent.com/render/math?math=N_\mathrm{full} = 20">: The number of training epochs used to train the model before applying adversarial training iterations.
- <img src="https://render.githubusercontent.com/render/math?math=N_\mathrm{adv} = 128">: The number of adversarial samples to search for.
- <img src="https://render.githubusercontent.com/render/math?math=N=128">: Size of training set. We used a mini-batch size of 8 during our stochastic gradient descent steps.
- <img src="https://render.githubusercontent.com/render/math?math=\gamma_\mathrm{recon} = 0.5">: The weight on control reconstruction when computing <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_\mathrm{recon}">.
- We use a sigmoidal function <img src="https://render.githubusercontent.com/render/math?math=\sigma_\mathrm{anneal}(i, l, u, b, c) = l + (u-l) \exp \left( \frac{i - bc}{b + i - bc}  \right)">
to characterize the annealing schedule various hyperparameters. Let <img src="https://render.githubusercontent.com/render/math?math=i"> denote the current training epoch, and <img src="https://render.githubusercontent.com/render/math?math=N"> denote the number of iterations for that training round (either <img src="https://render.githubusercontent.com/render/math?math=N_\mathrm{full}=200"> or <img src="https://render.githubusercontent.com/render/math?math=N_\mathrm{mini}=20">). Let <img src="https://render.githubusercontent.com/render/math?math=l"> and <img src="https://render.githubusercontent.com/render/math?math=u"> denote the smallest and largest value that the annealed hyperparameter can take.
For the full training phase where <img src="https://render.githubusercontent.com/render/math?math=N=N_\mathrm{full}">, <img src="https://render.githubusercontent.com/render/math?math=b = \frac{8N}{1000}"> and <img src="https://render.githubusercontent.com/render/math?math=c=6">:
    - <img src="https://render.githubusercontent.com/render/math?math=p_\mathrm{LSTM}(i) = \sigma_\mathrm{anneal}(i, 0.1, 1.0, b, c)"> 
    - <img src="https://render.githubusercontent.com/render/math?math=\gamma_\mathrm{STL}(i) = \sigma_\mathrm{anneal}(i, 0.1, \gamma_\mathrm{STL}, b, c)"> 
    - <img src="https://render.githubusercontent.com/render/math?math=\beta_\mathrm{STL}(i) = \sigma_\mathrm{anneal}(i, 0.1, 50, b, c)"> 
For the training phase during the adversarial training iterations where <img src="https://render.githubusercontent.com/render/math?math=N=N_\mathrm{mini}">, <img src="https://render.githubusercontent.com/render/math?math=b = \frac{8N}{1000}"> and <img src="https://render.githubusercontent.com/render/math?math=c=6">:
    - <img src="https://render.githubusercontent.com/render/math?math=p_\mathrm{LSTM}(i) = \sigma_\mathrm{anneal}(i, 0.8, 1.0, b, c)"> 
    - <img src="https://render.githubusercontent.com/render/math?math=\gamma_\mathrm{STL}(i) = \frac{3}{2}\gamma_\mathrm{STL}"> 
    - <img src="https://render.githubusercontent.com/render/math?math=\beta_\mathrm{STL}(i) = \sigma_\mathrm{anneal}(i, 20, 50, b, c)"> 


See `ctrl_syn/src/learning.py` see the neural network architecture.

To train a model, run
`python3 run_cnn.py --iter_max 200 --lstm_dim 256 --device cuda --dropout 0.2 --weight_ctrl 0.5 --weight_recon 0.7 --weight_stl -1 --teacher_training -1 --mode adv_training_iteration_rapid --type drive --stl_scale -1 --status new --stl_max 0.5 --stl_min 0.1 --scale_min 0.1 --scale_max 50.0 --trainset_size 128 --evalset_size 32 --number 0 --action create --adv_iter_max 100 --run 0 --expert_mini_bs 1`

