import tensorflow as tf
import argparse
import os
import time
import pickle
import ipdb

from model import SocialModel
from utils import SocialDataLoader
from grid import getSequenceGridMask


def main():
    parser = argparse.ArgumentParser()
    # RNN大小参数（输出/隐藏状态的维度）
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    #只实现了一个层
    # 层数
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # 类型
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter  batch size
    parser.add_argument('--batch_size', type=int, default=128,   #default 16
                        help='minibatch size')
    # 每个序列的长度
    parser.add_argument('--seq_length', type=int, default=12, #default 12
                        help='RNN sequence length')
    #  epoch数目
    parser.add_argument('--num_epochs', type=int, default=10,    #before the default was 50
                        help='number of epochs')
    # 存模型的频率
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')

    # （对模型没用？）
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # 学习率
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # 学习率削减
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')

    # 单层模型没有用到这个参数
    parser.add_argument('--keep_prob', type=float, default=1,
                        help='dropout keep probability')
    #  embeddings 
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # 无用参数
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # 相邻车辆数目N
    parser.add_argument('--grid_size', type=int, default=8,
                        help='Grid size of the social grid')
    # 每帧最大车辆数目
    parser.add_argument('--maxNumPeds', type=int, default=55,
                        help='Maximum Number of Pedestrians')
    # 用于测试的数据集
    parser.add_argument('--leaveDataset', type=int, default=1,
                        help='The dataset index to be left out in training')
    # (L2) Lambda正则化参数
    parser.add_argument('--lambda_param', type=float, default=0.001,
                        help='L2 regularization parameter')
    args = parser.parse_args(args=[])
    train(args)


def train(args):
    datasets = list(range(5))
    #从数据集中删除leaveDataset
    datasets.remove(args.leaveDataset)

    # Create the SocialDataLoader object
    data_loader = SocialDataLoader(args.batch_size, args.seq_length, args.maxNumPeds, datasets, forcePreProcess=True, infer=False)

    # 日志目录
    log_directory = 'log/'
    log_directory += str(args.leaveDataset) + '/'

    # Logging files
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    #存储目录
    save_directory = 'save/'
    save_directory += str(args.leaveDataset) + '/'
    

    with open(os.path.join(save_directory, 'social_config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Create a SocialModel object with the arguments
    model = SocialModel(args)

    with tf.Session() as sess:
        # Initialize all variables in the graph
        sess.run(tf.global_variables_initializer())
        # Initialize a saver that saves all the variables in the graph
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        # summary_writer = tf.train.SummaryWriter('/tmp/lstm/logs', graph_def=sess.graph_def)
        print ('Training begin')
        best_val_loss = 100
        best_epoch = 0

        # For each epoch
        for e in range(args.num_epochs):
            # Assign the learning rate value for this epoch
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # Reset the data pointers in the data_loader
            data_loader.reset_batch_pointer(valid=False)

            loss_epoch = 0

            # For each batch
            for b in range(data_loader.num_batches):
   
                start = time.time()

                # Get the source, target and dataset data for the next batch
                # 获取下一批的源，目标和数据集数据
                # x, y are input and target data which are lists containing numpy arrays of size seq_length x maxNumPeds x 3
                # x，y是输入和目标数据，它们是包含大小为seq_length x maxNumPeds x 3的numpy数组的列表
                # d is the list of dataset indices from which each batch is generated (used to differentiate between datasets)
                # d是生成每个批次的数据集索引列表（用于区分数据集）
                x, y, d = data_loader.next_batch()

                # variable to store the loss for this batch
                # 变量来存储此批次的损失
                loss_batch = 0

                # For each sequence in the batch
                for batch in range(data_loader.batch_size):
                    # x_batch, y_batch and d_batch contains the source, target and dataset index data for
                    # x_batch，y_batch和d_batch包含源，目标和数据集索引数据
                    # seq_length long consecutive frames in the dataset
                    # seq_length数据集中的长连续帧
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    # d_batch将是一个标量，用于标识从中提取此序列的数据集
                    x_batch, y_batch, d_batch = x[batch], y[batch], d[batch]

                    if d_batch == 0 and datasets[0] == 0:
                        dataset_data = [640, 480]
                    else:
                        dataset_data = [720, 576]

                    grid_batch = getSequenceGridMask(x_batch, dataset_data, args.neighborhood_size, args.grid_size)

                    # Feed the source, target data
                    # 提供源，目标数据
                    feed = {model.input_data: x_batch, model.target_data: y_batch, model.grid_data: grid_batch}
					# 运行模型，跑出来一个结果
                    train_loss, _ = sess.run([model.cost, model.train_op], feed)

                    loss_batch += train_loss

                end = time.time()
                loss_batch = loss_batch / data_loader.batch_size
                loss_epoch += loss_batch
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        loss_batch, end - start))

            loss_epoch /= data_loader.num_batches
            log_file_curve.write(str(e)+','+str(loss_epoch)+',')
            print ('*****************')

            # 验证模型
            data_loader.reset_batch_pointer(valid=True)
            loss_epoch = 0

            for b in range(data_loader.num_batches):

                # Get the source, target and dataset data for the next batch
                # x, y are input and target data which are lists containing numpy arrays of size seq_length x maxNumPeds x 3
                # d is the list of dataset indices from which each batch is generated (used to differentiate between datasets)
                x, y, d = data_loader.next_valid_batch()

                # variable to store the loss for this batch
                loss_batch = 0

                # For each sequence in the batch
                for batch in range(data_loader.batch_size):
                    # x_batch, y_batch and d_batch contains the source, target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    x_batch, y_batch, d_batch = x[batch], y[batch], d[batch]

                    if d_batch == 0 and datasets[0] == 0:
                        dataset_data = [640, 480]
                    else:
                        dataset_data = [720, 576]

                    grid_batch = getSequenceGridMask(x_batch, dataset_data, args.neighborhood_size, args.grid_size)

                    # Feed the source, target data
                    feed = {model.input_data: x_batch, model.target_data: y_batch, model.grid_data: grid_batch}

                    train_loss = sess.run(model.cost, feed)

                    loss_batch += train_loss

                loss_batch = loss_batch / data_loader.batch_size
                loss_epoch += loss_batch

            loss_epoch /= data_loader.valid_num_batches

            # Update best validation loss until now
            if loss_epoch < best_val_loss:
                best_val_loss = loss_epoch
                best_epoch = e

            print('(epoch {}), valid_loss = {:.3f}'.format(e, loss_epoch))
            print ('Best epoch', best_epoch, 'Best validation loss', best_val_loss)
            log_file_curve.write(str(loss_epoch)+'\n')
            print ('*****************')

            # Save the model after each epoch
            print ('Saving model')
            checkpoint_path = os.path.join(save_directory, 'social_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=e)
            print("model saved to {}".format(checkpoint_path))

        print ('Best epoch', best_epoch, 'Best validation loss', best_val_loss)
        log_file.write(str(best_epoch)+','+str(best_val_loss))

        # CLose logging files
        log_file.close()
        log_file_curve.close()


if __name__ == '__main__':
    main()
