'''
Handles processing the input and target data in batches and sequences

Modified by : Simone Zamboni
Date : 2018-01-10
'''

import os
import pickle
import numpy as np
import ipdb
import random

class SocialDataLoader():

    # Questo costruttore non e' stato modificato rispetto a quella in Social_LSTM
    # 此构造函数尚未针对Social_LSTM中的构造函数进行修改
    def __init__(self, batch_size=50, seq_length=5, maxNumPeds=200, datasets=[0, 1, 2, 3, 4], forcePreProcess=False, infer=False):

        # List of data directories where raw data resides (rispetto all'originale e' stato cambiato)
        # 原始数据所在的数据目录列表（相对于原始数据已更改）
        self.data_dirs = ['C:\\Users\\asus\\Desktop\\lstm\\基于递归神经网络的汽车轨迹预测\基于递归神经网络的汽车轨迹预测\\vehicle traj pre\\data\\test1',
                          'C:\\Users\\asus\\Desktop\\lstm\\基于递归神经网络的汽车轨迹预测\基于递归神经网络的汽车轨迹预测\\vehicle traj pre\\data\\test2',
                          'C:\\Users\\asus\\Desktop\\lstm\\基于递归神经网络的汽车轨迹预测\基于递归神经网络的汽车轨迹预测\\vehicle traj pre\\data\\test3',
                          'C:\\Users\\asus\\Desktop\\lstm\\基于递归神经网络的汽车轨迹预测\基于递归神经网络的汽车轨迹预测\\vehicle traj pre\\data\\test4',
                          'C:\\Users\\asus\\Desktop\\lstm\\基于递归神经网络的汽车轨迹预测\基于递归神经网络的汽车轨迹预测\\vehicle traj pre\\data\\test5']

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        self.infer = infer

        self.numDatasets = len(self.data_dirs)

        self.data_dir = 'C:\\Users\\asus\\Desktop\\lstm\\基于递归神经网络的汽车轨迹预测\基于递归神经网络的汽车轨迹预测\\vehicle traj pre\\data'
        self.maxNumPeds = maxNumPeds
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.val_fraction = 0.3
        self.takeOneInNFrames = 1
        data_file = os.path.join(self.data_dir, "social-trajectories.cpkl")

        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            self.frame_preprocess(self.used_data_dirs, data_file)

        self.load_preprocessed(data_file)

        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    #funzione principale modificata
    # 解释保存的内容：将以下变量写入保存在ckpl文件中
    # frameList_data [i] =第i个数据集的所有帧编号（如果数据集有700个帧，则将有一个700个元素的数组，范围从1到700
    # numpeds_data [i] [j] =当第i个数据集的第j帧中有行人的数量
    # all_frame_data[i][j]：第i个目录中所有第j帧的顺序为所有行人的列表：[id，x，y]，列表的长度为maxNumPeds，并包含之后的帧
    # 最后一个valid_frame_data
    # validframedata：与all_frame_data相同，仅作为仅具有验证帧的结构
    # goal [i] [j] =目标pawn的x和y坐标，视频i的id为j
    def frame_preprocess(self, data_dirs, data_file):

        all_frame_data = []
        valid_frame_data = []
        frameList_data = []
        numPeds_data = []

        dataset_index = 0

        frames = []  # 列出所有帧以all_frame_data的格式存储的列表
        all_peds = []  # array with the dimension of (numDirectory,b) with b the sum of each time all the pedestian appera 数组的维度为（numDirectory，b），b为每次所有pedestian appera的总和
        dataset_validation_index = []

        # For each dataset
        for directory in data_dirs:
            #
            file_path = os.path.join(directory, 'pixel_pos_interpolate.csv')
            # file_path = os.path.join(directory, 'roundabout_traj_ped_filtered.csv')

            data = np.genfromtxt(file_path, delimiter=',')

            #changed
            # data = np.transpose(data)
            # data = (data[4, 1:], data[1, 1:], data[6, 1:], data[7, 1:])
            # y = 2 * (data[3] - min(data[3])) / (max(data[3]) - min(data[3])) - 1
            # x = 2 * (data[2] - min(data[2])) / (max(data[2]) - min(data[2])) - 1
            # data = ((data[0]-1), data[1], x, y)
            # data=np.array(data)

            frameList = np.unique(data[0, :]).tolist() # unique 去除掉相同元素 组成的列表
            # Number of frames
            numFrames = int(len(frameList)/self.takeOneInNFrames)*self.takeOneInNFrames

            if self.infer:
                valid_numFrames = 0
            else:
                valid_numFrames = int((numFrames * self.val_fraction)/self.takeOneInNFrames)*self.takeOneInNFrames
    

            dataset_validation_index.append(valid_numFrames)

            frameList_data.append(frameList)   # now Framelist = [0,1,2,3...]

            numPeds_data.append([])
            all_peds.append([])
            # 生成一个 （训练集，最大行人数量，3）的一个全零数组
            all_frame_data.append(np.zeros( (int((numFrames - valid_numFrames)/self.takeOneInNFrames), self.maxNumPeds, 3) ) )
            # 生成验证机的数组
            valid_frame_data.append(np.zeros(  (int(valid_numFrames/self.takeOneInNFrames), self.maxNumPeds, 3) ) )
            #生成所有数据的数组
            frames.append(np.zeros((numFrames, self.maxNumPeds, 3)))

            ind = 0
            while ind < numFrames:
                frame = frameList[ind]
                pedsInFrame = data[:, data[0, :] == frame]  # 取出相应frame的数据
                pedsList = pedsInFrame[1, :].tolist() # 取出 相应frame的行人的id

                numPeds_data[dataset_index].append(len(pedsList))

                pedsWithPos = []

                for ped in pedsList:

                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    pedsWithPos.append([ped, current_x, current_y])
                    all_peds[dataset_index].append((ped))

                if (ind >= valid_numFrames) or (self.infer):
                    all_frame_data[dataset_index][int((ind - valid_numFrames)/self.takeOneInNFrames), 0:len(pedsList), :] = np.array(pedsWithPos)
                else:
                    valid_frame_data[dataset_index][int(ind/self.takeOneInNFrames), 0:len(pedsList), :] = np.array(pedsWithPos)

                frames[dataset_index][ind, 0:len(pedsList), :] = np.array(pedsWithPos)
                ind += self.takeOneInNFrames

            dataset_index += 1

        #passiamo ora al calcolo del "goal" di ogni pedone
        # 让我们继续计算每个行人的“目标”

        unique_all_peds = [] #array che conterra' per ogni video il numero totale di pedoni
        # #array将包含每个视频的行人总数

        #Ciclo che per ogni video salva nell'array il numero di pedoni di quel video
        # While为每个视频保存数组中该视频的行人数
        dir = 0
        while dir < len(data_dirs):
            unique_all_peds.append(np.unique(all_peds[dir]))
            dir += 1

        goal = []  # array contenente l'obbiettivo di ogni pedone 包含每个棋子的目标阵列
        # questo array ha dimensioni: (num_video, num_pedestrian_for_that_dir, 2), 2 si riferisce alla x e alla y del goal di ogni pedone
        # 这个数组有尺寸：(num_video, num_pedestrian_for_that_dir, 2) 2指的是每个棋子的目标的x和y
        # Inizializzazione dell'array goal con tutti 0 使用全0来初始化目标数组
        dir = 0
        while dir < len(data_dirs):
            goal.append([])
            ped = 0
            #sembra che il valore len(unique_all_peds[dir]) non ritorni il numero di pedoni esatto in un video
            # e se non ci aggiungessimo una valore abbastanza alto darebbe errore.
            # Si e' quindi deciso di aggiungere una valore molto alto arbiratio per evitare errori, questa e' un
            # punto del codice che potrebbe assolutamente essere migliorato.
            #似乎值len（unique_all_peds[dir]）没有返回视频中行人的确切数量
            #并且如果我们没有添加足够高的值，它将给出错误。
            #因此决定添加一个非常高的值以避免错误，这是一个
            #代码点，绝对可以改进。
            while ped <= len(unique_all_peds[dir]) + 1000:
                goal[dir].append([0, 0])
                ped += 1
            dir += 1

        # per ogni frame dei video aggiornare l'ultima posizione conosciuta di quel pedone
        # 针对视频的每帧更新的棋子的最后已知位置
        dir = 0
        while dir < len(frames):
            frame = 0
            while frame < len(frames[dir]):
                ped_n = 0
                #per ogni pedone in ogni frame di ogni video
                # 每一个行人在每个视频的每一帧
                while ped_n < len(frames[dir][frame]):
                    ped_id = int(frames[dir][frame][ped_n][0]) #ricaviamo l'id del pedone attuale 当前行人的id
                    goal[dir][ped_id][0] = frames[dir][frame][ped_n][1] #nell'array goal mettiamo le sue coordinate attuali
                    goal[dir][ped_id][1] = frames[dir][frame][ped_n][2] # 在目标数组中我们放置它的当前坐标
                    ped_n += 1
                frame += 1
            dir += 1

        #此时，在每个帧中每个行人的目标数组中，我们应该具有最后的已知位置
        #解释保存的内容：
        # frameList_data [i] =第i个数据集的所有帧编号（如果数据集有700个帧，则将有一个700个元素的数组，范围从1到700
        # numpeds_data [i] [j] =当第i个数据集的第j帧中有行人的数量
        # all_frame_data[i][j]：第i个目录中所有第j帧的顺序为所有行人的列表：[id，x，y]，列表的长度为maxNumPeds，并包含之后的帧
        #最后一个valid_frame_data
        #validframedata：与all_frame_data相同，仅作为仅具有验证帧的结构
        # goal [i] [j] =目标pawn的x和y坐标，视频i的id为j

        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_frame_data, goal), f, protocol=2)
        f.close()

    #funzione modificata per caricare anche il goal
    def load_preprocessed(self, data_file):

        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_data = self.raw_data[3]
        self.goals = self.raw_data[4] #prendo anche il goal dal file salvato
        counter = 0
        valid_counter = 0

        for dataset in range(len(self.data)):
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            print ('Training data from dataset', dataset, ':', len(all_frame_data))
            print ('Validation data from dataset', dataset, ':', len(valid_frame_data))
            counter += int(len(all_frame_data) / (self.seq_length+2))
            valid_counter += int(len(valid_frame_data) / (self.seq_length+2))

        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)
        #self.num_batches = self.num_batches * 2


    #funzione modificata per fare in modo che nei dati del batch ci sia anche il goal del pedone
    # 修改功能使行人的目标也出现在批处理数据中
    def next_batch(self, randomUpdate=True):
        x_batch = []
        y_batch = []
        d = []
        i = 0

        while i < self.batch_size:
            frame_data = self.data[self.dataset_pointer]

            idx = self.frame_pointer
            # 循环每个帧
            if idx + self.seq_length < frame_data.shape[0]:

                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]

                #list of the ID of all the pedestrian in the current batch
                # list中当前批次中所有行人的ID
                pedID_list = np.unique(seq_frame_data[:, :, 0])

                # Number of unique peds the current batch
                numUniquePeds = pedID_list.shape[0]

                # sia sourceData che targetData sono stati ampliati da 3 a 5
                # #sourceData和targetData都已从3扩展到5 (5,max,5)
                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 5))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 5))

                #per ogni frame della sequenza
                #对序列的每个帧
                for seq in range(self.seq_length):
                    # frame attuale (ssqe_frame_data) e successivo (tseq_frame_data)
                    # 当前帧（ssqe_frame_data）和下一个（tseq_frame_data）
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]

                    #per tutti i pedoni nel frame
                    for ped in range(numUniquePeds):
                        #prendere il pedID
                        pedID = pedID_list[ped]

                        #se il pedone non esiste andare avanti al prossimo giro
                        # ＃如果行人不存在则继续下一轮

                        if pedID == 0:
                            continue
                        else:
                            tped = [] #target data per questo pedone
                            sped = [] #sequence data per questo pedone

                            #prendere la posizione del pedone nel frame
                            # 取走框架中行人的位置
                            temp_sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]

                            #se quel pedone e' presente nel frame, cioe' se ha una posizione(ed e' quindi salvata in temp_sped) allora si va
                            #＃如果该行人存在于框架中，即它是否有位置（因此保存在temp_sped中）然后它继续
                            if len(temp_sped) > 0 :
                                #aggiungere ai dati di input del pedone la posizione del pedone
                                iter= 0
                                while iter < len (temp_sped[0]):
                                    sped.append(temp_sped[0][iter])
                                    iter+=1

                                #e aggiungere i dati di input del pedone le coordinate del goal del pedone
                                # ＃并将行人输入数据添加到行人的目标坐标
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            #array temporameo che contiene i dati della posizione futura del pedone
                            # #array temporameo，其中包含行人未来位置的数据
                            temp_tped = tseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                            #se quel pedone ha dati target, cioe' se ha una posizione anche nel frame successivo (questa quindi sara' salvata in temp_sped)
                            # ＃如果该pawn有目标数据，即它是否在下一帧中有一个位置（这将保存在temp_sped中）
                            if(len(temp_tped) > 0) :
                                iter = 0
                                #aggiungere ai dati target di quel pedone la sua posizione futura
                                while iter < len(temp_tped[0]):
                                    tped.append(temp_tped[0][iter])
                                    iter += 1
                                # e aggiungere i dati target di quel pedone anche le coordinate del goal
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            #se sono state inserite delle informazioni in sped e tped allora vengono aggiunti a sourceData e targetData
                            #如果已在发送中输入信息并将其添加到源数据和目标数据中
                            if len(sped) > 2:
                                sourceData[seq, ped, :] = sped
                            if len(tped) > 2:
                                targetData[seq, ped, :] = tped

                x_batch.append(sourceData)
                y_batch.append(targetData)

                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1
            else:
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, d

    # funzione modificata per fare in modo che nei dati del batch ci sia anche il goal del pedone, modifiche praticamente identiche alla funzione precedente
    def next_valid_batch(self, randomUpdate=True):
        x_batch = []
        y_batch = []
        d = []
        i = 0
        while i < self.batch_size:
            frame_data = self.valid_data[self.valid_dataset_pointer]
            idx = self.valid_frame_pointer

            if idx + self.seq_length < frame_data.shape[0]:
                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]

                # list of the ID of all the pedestrian in the current batch
                pedID_list = np.unique(seq_frame_data[:, :, 0])
                # Number of unique peds the current batch
                numUniquePeds = pedID_list.shape[0]

                # sia sourceData che targetData sono stati ampliati da 3 a 5
                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 5))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 5))

                # per ogni frame della sequenza
                for seq in range(self.seq_length):
                    # frame attuale (ssqe_frame_data) e successivo (tseq_frame_data)
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]

                    # per tutti i pedoni nel frame
                    for ped in range(numUniquePeds):
                        pedID = pedID_list[ped] #prendere il pedID

                        # se il pedone non esiste andare avanti al prossimo ciclo
                        if pedID == 0:
                            continue
                        else:
                            tped = [] #target data per questo pedone
                            sped = [] #sequence data per questo pedone

                            # array che contiene la posizione del pedone nel frame
                            temp_sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]

                            # se quel pedone e' presente nel frame, cioe' se ha una posizione(ed e' quindi salvata in temp_sped) allora si va avanti
                            if(len(temp_sped) > 0):
                                # aggiungere ai dati di input del pedone la posizione del pedone
                                iter = 0
                                while iter < len(temp_sped[0]):
                                    sped.append(temp_sped[0][iter])
                                    iter += 1

                                # e aggiungere i dati di input del pedone le coordinate del goal del pedone
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            # array che contiene i dati della posizione futura del pedone
                            temp_tped = (tseq_frame_data[tseq_frame_data[:, 0] == pedID, :])

                            # se quel pedone ha dati target, cioe' se ha una posizione anche nel frame successivo (questa quindi sara' salvata in temp_sped)
                            if(len(temp_tped) > 0) :
                                # aggiungere ai dati target di quel pedone la sua posizione futura
                                iter = 0
                                while iter < len(temp_tped[0]):
                                    tped.append(temp_tped[0][iter])
                                    iter += 1

                                # e aggiungere i dati target di quel pedone anche le coordinate del goal
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            # se sono state inserite delle informazioni in sped e tped allora vengono aggiunti a sourceData e targetData
                            if len(sped) > 2:
                                sourceData[seq, ped, : ] = sped
                            if len(tped) > 2:
                                targetData[seq, ped, :] = tped

                x_batch.append(sourceData)
                y_batch.append(targetData)

                # Advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1
            else:
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d

    #funzione non modificata
    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0

    #funzione non modificata
    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0

