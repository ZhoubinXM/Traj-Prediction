'''
Modello per il Social_LSTM che comprende il goal e l'array sociale

Modified by: Simone Zamboni
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell
from grid import getSequenceGridMask
import pdb


class SocialModel():

    #costruttore modificato
    def __init__(self, args, infer=False):

        if infer:
            args.batch_size = 1
            args.seq_length = 1

        self.args = args
        self.infer = infer
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size

        self.maxNumPeds = args.maxNumPeds

        with tf.name_scope("LSTM_cell"):
            cell = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)

        #输入数据已被修改。它现在包含序列的每个帧的行人数据以及帧中每个可能的行人
        #行人数据按此顺序排列：[ID，x_ current，y_ current，goal_x，goal_y]
        self.input_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, 5], name="input_data")

        # 目标数据已被修改。它现在包含序列的每个帧的行人数据以及帧中每个可能的行人
        #行人数据按此顺序排列：[ID，x_futura，y_futura，goal_x，goal_y]
        self.target_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, 5], name="target_data")

        # grid_data passa da dimensione SL X MNP X MNP x grid_size*grid_size a SL X MNP X grid_size*2
        # grid_data从SL维度X MNP X MNP x grid_size * grid_size更改为SL X MNP X grid_size * 2
        self.grid_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, args.grid_size*2], name="grid_data")

        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        self.output_size = 5

        self.define_embedding_and_output_layers(args)

        with tf.variable_scope("LSTM_states"):
            self.LSTM_states = tf.zeros([args.maxNumPeds, cell.state_size], name="LSTM_states")
            self.initial_states = tf.split(self.LSTM_states, args.maxNumPeds, 0 )

        with tf.variable_scope("Hidden_states"):
            self.output_states = tf.split(tf.zeros([args.maxNumPeds, cell.output_size]), args.maxNumPeds,0 )

        with tf.name_scope("frame_data_tensors"):
            frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.input_data, args.seq_length, 0)]

        with tf.name_scope("frame_target_data_tensors"):

            frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(self.target_data, args.seq_length, 0)]

        with tf.name_scope("grid_frame_data_tensors"):
            # grid frame data viene diviso diventanto una lista di seq_length array ognuno contenente le informazioni per quel frame
            #网格帧数据被分成一个seq_length数组列表，每个数组包含该帧的信息
            grid_frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.grid_data, args.seq_length, 0)]

        with tf.name_scope("Cost_related_stuff"):
            self.cost = tf.constant(0.0, name="cost")
            self.counter = tf.constant(0.0, name="counter")
            self.increment = tf.constant(1.0, name="increment")

        with tf.name_scope("Distribution_parameters_stuff"):
            self.initial_output = tf.split(tf.zeros([args.maxNumPeds, self.output_size]), args.maxNumPeds, 0 )

        with tf.name_scope("Non_existent_ped_stuff"):
            nonexistent_ped = tf.constant(0.0, name="zero_ped")
        #对于某个序列进行操作
        for seq, frame in enumerate(frame_data):
            print ("Frame number", seq)

            current_frame_data = frame  # MNP x 5 tensor
            # non serve piu' la funzione getSocialTensor perche' e' gia' tutto dentro grid_frame_data
            # 你不再需要getSocialTensor函数，因为它全部在grid_frame_data中
            # infatti basta prendere il grid_frame_data del frame corrente e si ha gia' l'array sociale di ogni pedone nel frame
            # 实际上，只需获取当前帧的grid_frame_data，您就已经拥有了帧中每个行人的社交数组
            current_grid_frame_data = grid_frame_data[seq]

            for ped in range(args.maxNumPeds):
                print ("Vehicle Number", ped)

                pedID = current_frame_data[ped, 0]

                with tf.name_scope("extract_input_ped"):
                    # Estrarre le coordinate x e y del pedone attuale, insieme alle coordinate x e y del suo goal
                    #提取当前行人x和y坐标，以及其目标的x和y坐标
                    self.spatial_input = tf.slice(current_frame_data, [ped, 1], [1, 4])

                    temp = [] #array temporaneo per poter mettere i dati in current_grid_data nella giusta forma
                    #临时数组，以正确的形式将数据放入current_grid_data中
                    temp.append((current_grid_frame_data[ped]))
                    self.tensor_input = temp # Tensor di dimensione 1 X grid_size*2
                    #张量的尺寸 1x grid_size*2

                with tf.name_scope("embeddings_operations"):
                    # Da notare che embedding_w e' stato ampliato da 2 x embedding_size a 4 x embedding_size
                    # 请注意，embedding_w已从2 x embedding_size扩展为4 x embedding_size
                    embedded_spatial_input = tf.nn.relu(tf.nn.xw_plus_b(self.spatial_input, self.embedding_w, self.embedding_b))
                    # Da notare che embedding_t_w e embedding_t_b hanno cambiato dimensioni per rispecchiare le nuove dimensioni della griglia
                    # 请注意，embedding_t_w和embedding_t_b已更改大小以反映新的网格大小
                    embedded_tensor_input = tf.nn.relu(tf.nn.xw_plus_b(self.tensor_input, self.embedding_t_w, self.embedding_t_b))

                with tf.name_scope("concatenate_embeddings"):

                    complete_input = tf.concat([embedded_spatial_input, embedded_tensor_input], 1)

                # One step of LSTM
                with tf.variable_scope("LSTM") as scope:
                    if seq > 0 or ped > 0:
                        scope.reuse_variables()
                    self.output_states[ped], self.initial_states[ped] = cell(complete_input, self.initial_states[ped])


                with tf.name_scope("output_linear_layer"):
                    self.initial_output[ped] = tf.nn.xw_plus_b(self.output_states[ped], self.output_w, self.output_b)

                with tf.name_scope("extract_target_ped"):
                    # x_data e y_data sono rimasti invariati, mentre target_pedID ora contiene anche il goal
                    # x_data和y_data保持不变，而target_pedID现在也包含目标
                    [x_data, y_data] = tf.split(tf.slice(frame_target_data[seq], [ped, 1], [1, 2]), 2, 1)
                    target_pedID = frame_target_data[seq][ped, 0]

                with tf.name_scope("get_coef"):
                    [o_mux, o_muy, o_sx, o_sy, o_corr] = self.get_coef(self.initial_output[ped])

                with tf.name_scope("calculate_loss"):
                    lossfunc = self.get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

                with tf.name_scope("increment_cost"):
                    self.cost = tf.where(
                        tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)),
                        self.cost, tf.add(self.cost, lossfunc))
                    self.counter = tf.where(
                        tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)),
                        self.counter, tf.add(self.counter, self.increment))

        with tf.name_scope("mean_cost"):
            self.cost = tf.div(self.cost, self.counter)

        tvars = tf.trainable_variables()

        l2 = args.lambda_param*sum(tf.nn.l2_loss(tvar) for tvar in tvars)
        self.cost = self.cost + l2

        self.final_states = tf.concat(self.initial_states, 0)

        self.final_output = self.initial_output

        self.gradients = tf.gradients(self.cost, tvars)

        grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        optimizer = tf.train.RMSPropOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def define_embedding_and_output_layers(self, args):
        # ora embedding_w non e' piu' di dimensioni 2 X embedding_size ma 4 X embedding_size, cosi' puo' essere moltiplicato
        # 现在embedding_w不再是2 X embedding_size而是4 X embedding_size，所以它可以相乘
        # per l'array contenente le 4 coordinate che sono x attuale, y attuale, x del goal e y del goal
        # 对于包含4个坐标的数组，这些坐标是x current，y current，x的目标和y的目标
        with tf.variable_scope("coordinate_embedding"):
            self.embedding_w = tf.get_variable("embedding_w", [4, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding_b = tf.get_variable("embedding_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))

        # embedding_t_w passa da grid_size*grid_size*rnn_size X embedding_size a grid_size*2 X 1
        # embedding_t_b e' passato da dimensione embedding_size a 1
        # al posto del valore 1 poteva essere usato il valore embedding_size, e' stato usato il valore 1 perche' dava risultati leggermente
        # migliori. Si consiglia di fare delle prove con diversi valori per capire come meglio impostare questo parametro
        # embedding_t_w 更改为 a grid_size*2 X 1
        # embedding_t_b从embedding_size更改为1
        # 可以使用值embedding_size代替值1，使用值1，因为它给出了轻微的结果
        # 最好的。建议使用不同的值进行测试，以了解如何最好地设置此参数
        with tf.variable_scope("tensor_embedding"):
            self.embedding_t_w = tf.get_variable("embedding_t_w", [args.grid_size*2,args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding_t_b = tf.get_variable("embedding_t_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))

        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", [args.rnn_size, self.output_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.output_b = tf.get_variable("output_b", [self.output_size], initializer=tf.constant_initializer(0.1))

    #funzione non modificata
    def tf_2d_normal(self, x, y, mux, muy, sx, sy, rho):
        '''
        Function that implements the PDF of a 2D normal distribution 二维正态分布
        params:
        x : input x points
        y : input y points
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x    x分布的标准偏差
        sy : std dev of the distribution in y    y分布的标准偏差
        rho : Correlation factor of the distribution
        '''
        # eq 3 in the paper
        # and eq 24 & 25 in Graves (2013)
        # Calculate (x - mux) and (y-muy)
        normx = tf.subtract(x, mux) #was normx = tf.sub(x, mux)
        normy = tf.subtract(y, muy) #was normy = tf.sub(y, muy)
        # Calculate sx*sy
        sxsy = tf.multiply(sx, sy) #was sxsy = tf.mul(sx, sy)
        # Calculate the exponential factor 计算指数因子
        z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2 * tf.div(tf.multiply(rho, tf.multiply(normx, normy)),
                                                                                     sxsy)
        #was z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.mul(rho, tf.mul(normx, normy)), sxsy)
        negRho = 1 - tf.square(rho)
        # Numerator
        result = tf.exp(tf.div(-z, 2*negRho))
        # Normalization constant 归一化常数
        denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho)) #was denom = 2 * np.pi * tf.mul(sxsy, tf.sqrt(negRho))
        # Final PDF calculation
        result = tf.div(result, denom)
        return result

    # funzione non modificata
    def get_lossfunc(self, z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
        '''
        Function to calculate given a 2D distribution over x and y, and target data
        of observed x and y points用于计算给定的x和y上的2D分布以及观察到的x和y点的目标数据的函数
        params:
        z_mux : mean of the distribution in x  x的分布均值
        z_muy : mean of the distribution in y  Y的分布均值
        z_sx : std dev of the distribution in x
        z_sy : std dev of the distribution in y
        z_rho : Correlation factor of the distribution  分布的相关因素
        x_data : target x points
        y_data : target y points
        '''
        # step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

        # Calculate the PDF of the data w.r.t to the distribution
        result0 = self.tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

        # For numerical stability purposes
        epsilon = 1e-20

        # Apply the log operation
        result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

        # Sum up all log probabilities for each data point
        return tf.reduce_sum(result1)

    # funzione non modificata
    def get_coef(self, output):
        # eq 20 -> 22 of Graves (2013)

        z = output
        # Split the output into 5 parts corresponding to means, std devs and corr
        z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, 1) #was z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(1, 5, z)

        # The output must be exponentiated for the std devs
        z_sx = tf.exp(z_sx)
        z_sy = tf.exp(z_sy)
        # Tanh applied to keep it in the range [-1, 1]
        z_corr = tf.tanh(z_corr)

        return [z_mux, z_muy, z_sx, z_sy, z_corr]

    # funzione non modificata
    def sample_gaussian_2d(self, mux, muy, sx, sy, rho):
        '''
        Function to sample a point from a given 2D normal distribution
        params:
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        '''
        # Extract mean
        mean = [mux, muy]
        # Extract covariance matrix
        cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
        # Sample a point from the multivariate normal distribution
        x = np.random.multivariate_normal(mean, cov, 1)

        # Modifica di SIMONE per non utilizzare un numero random per decidere la posizione futura del pedone:
        #修改SIMONE不要使用随机数来决定行人的未来位置：
        return mux, muy  # era return x[0][0], x[0][1]

    #funzione modificata
    def sample(self, sess, traj, grid, dimensions, true_traj, num=10):
        states = sess.run(self.LSTM_states)

        for index, frame in enumerate(traj[:-1]):
            # data ora ha dimensione 1 X maxNumPeds X 5 invece che X 3 cosi' da poter contenere anche il goal
            # 数据现在的大小为1 X maxNumPeds X 5而不是X 3，因此它也可以包含目标
            # lo stesso vale per target_data
            # target_data也是如此
            data = np.reshape(frame, (1, self.maxNumPeds, 5))
            target_data = np.reshape(traj[index+1], (1, self.maxNumPeds, 5))
            # grid_data ha dimensioni 1 X NMP X grid_size*2 a differenza di prima che aveva dimensioni 1 X MNP X MNP X grid_size*grid_size
            grid_data = np.reshape(grid[index, :], (1, self.maxNumPeds, self.grid_size*2))

            feed = {self.input_data: data, self.LSTM_states: states, self.grid_data: grid_data, self.target_data: target_data}

            [states, cost] = sess.run([self.final_states, self.cost], feed)

        ret = traj

        last_frame = traj[-1]

        # come prima sono stati ampliati anche prev_data e prev_grid_data da 3 a 5 per potre contenere anche il goal
        # 和以前一样，prev_data和prev_grid_data也从3扩展到5，甚至可以包含目标
        prev_data = np.reshape(last_frame, (1, self.maxNumPeds, 5))
        # come grid_data e' stata modificata la dimensione anche di prev_grid_data
        # 作为grid_data，prev_grid_data的大小也已更改
        prev_grid_data = np.reshape(grid[-1], (1,self.maxNumPeds, self.grid_size*2))

        prev_target_data = np.reshape(true_traj[traj.shape[0]], (1, self.maxNumPeds, 5))

        for t in range(num):

            feed = {self.input_data: prev_data, self.LSTM_states: states, self.grid_data: prev_grid_data, self.target_data: prev_target_data}
            [output, states, cost] = sess.run([self.final_output, self.final_states, self.cost], feed)

            # le nuove posizioni hanno dimensione 1 X 5 invece che 1 X 3 per fare spazio al goal
            newpos = np.zeros((1, self.maxNumPeds, 5))
            for pedindex, pedoutput in enumerate(output):
                [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(pedoutput[0], 5, 0)
                mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])

                next_x, next_y = self.sample_gaussian_2d(mux, muy, sx, sy, corr)

                newpos[0, pedindex, :] = [prev_data[0, pedindex, 0], next_x, next_y, 0, 0]
            ret = np.vstack((ret, newpos))
            prev_data = newpos
            prev_grid_data = getSequenceGridMask(prev_data, dimensions, self.args.neighborhood_size, self.grid_size)
            if t != num - 1:
                prev_target_data = np.reshape(true_traj[traj.shape[0] + t + 1], (1, self.maxNumPeds, 5))

        return ret
