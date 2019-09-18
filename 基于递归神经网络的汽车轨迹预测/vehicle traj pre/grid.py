'''
Funzioni che calcolano l'array sociale

Modified by: Simone Zamboni
'''
import numpy as np
import math


def getGridMask(frame, dimensions, neighborhood_size, grid_size):
    '''
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : inutile, tenuto per semplificare l'implementaizione e modificare solamente questo file
    neighborhood_size : inutile, tenuto per semplificare l'implementaizione e modificare solamente questo file
    grid_size : Quanti pedoni saranno presenti nell'array
    参数：
    框架：这将是一个MNP x 3矩阵，每行都是[pedID，x，y]
    维度：无用，需要简化实现并且只修改此文件
    neighborhood_size：无用，需要简化实现和修改只有这个文件
    grid_size：数组中会有多少行人
    '''

    # Estrarre e salvare il massimo numero di pedoni nel frame
    # 提取并保存帧中的最大行人数(矩阵的行数）
    mnp = frame.shape[0]

    #Array che contiene per ogni pedone nel frame il suo array sociale
    #每个阵列包含在帧的行人社会阵列
    my_array = np.zeros((mnp,grid_size*2))

    #Per ogni pedone nell'array viene creato il suo array sociale
    #对于数组中的每个行人，都会创建其社交数组
    for pedindex in range(mnp):

        #se il pedone non esiste (quindi ha ID = 0) si passa al prossimo ciclo
        #如果行人不存在（因此ID = 0），它将进入下一个循环
        if(frame[pedindex,0] == 0):
           continue

        #prendere la posizione attuale del pedone preso in considerazione
        #考虑当前的行人位置
        current_x,current_y =  frame[pedindex, 1], frame[pedindex, 2]
        other_peds_with_position = []

        #per ogni pedone nel frame
        #对于框架中的每个行人
        for otherpedindex in range(mnp):

            #se il pedone non esiste (quindi ha ID = 0) si passa al prossimo ciclo
            #如果行人不存在（因此ID = 0），它将进入下一个循环
            if frame[otherpedindex, 0] == 0:
                continue

            #otherpedindex e' uguale a pedindex e quindi sono lo stesso pedone, quindi si passa al prossimo ciclo, perche' un pedone non puo' essere presente nel suo stesso array
            #其他行人与行人相同，因此是同一个行人，所以我们传递到下一个循环，因为行人不能出现在它自己的数组中
            if frame[otherpedindex, 0] == frame[pedindex, 0]:
                continue

            #calcolare la distanza dal pedone attuale al pedone otherpedindex
            #计算从当前行人到另一个行人的距离
            current_distance = math.sqrt( math.pow((current_x - frame[otherpedindex][1]),2) + math.pow((current_y - frame[otherpedindex][2]),2)  )

            #salvare nell'array other_peds_with_position ID,x,y,distanza_dal_pedone_attuale del pedone otherpedindex
            other_peds_with_position.append( [frame[otherpedindex][0],frame[otherpedindex][1],frame[otherpedindex][2],current_distance])

        #ora abbiamo un array contenente ID,x,y e distanza di tutti gli altri pedoni validi
        #现在我们有一个包含ID，x，y和所有其他有效行人距离的数组
        #se dopo aver controllato tutto il frame non vi sono altri pedoni all'infuori di quello attuale un pedone finto viene inserito
        #如果在检查整个框架之后没有其他行人，除了当前的行人，插入假人
        if (len(other_peds_with_position) == 0):
            #questo pedone avra' coordinate x-2 e y-2 rispetto al pedone attuale, cosi' da essere molto lontano
            #这个假人将具有相对于当前行人的x-2和y-2坐标，因此距离很远
            other_peds_with_position.append([0, frame[pedindex,1]-2, frame[pedindex,1]-2, 2,828427125])
            
        #numero di quanti altri pedoni sono stati trovati
        # 已找到多少其他行人
        num_other_peds = len(other_peds_with_position)

        #scorro l'array sociale del pedone attuale e lo riempio
        #我滚动当前行人的社交数组并填充它
        j = 0 #indica il pedone j-esimo nell'array sociale, quindi j*2 indica la coordinata x e j*2+1 indica la coordinata y del j-esimo pedone nell'array sociale
        #表示社交数组中的第j个行人，然后j * 2表示x坐标，j * 2 + 1表示y坐标
        while j < len(my_array[pedindex]):
            x = 0 #usato per scorrerei pedoni nell'array other_peds_with_position 用于在other_peds_with_position数组中滚动行人

            # array che contiene in prima posizione la distanza minima trovata e in seconda la posizione del pedone con la minima distanza nell'array other_peds_with_position
            #在第一个位置包含最小距离，在第二个位置包含数组中最小距离的行人的ID
            min_distance = [1000000,0]
            update = False

            # per ogni pedone nell'array other_peds_with_position si cerca il piu' vicino al pedone attuale
            # 对于数组other_peds_with_position中的每个行人，我们寻找最接近当前行人的东西
            while x < len(other_peds_with_position):
                # se il pedone x ha la distanza finora minore salvo la distanza in min_distance[0] e la posizione del pedone in min_distance[1]
                #如果行人 x到目前为止的距离最短，除了min_distance [0]中的距离和min_distance [1]中的pawn位置
                if(other_peds_with_position[x][3] < min_distance[0]):
                    min_distance[0] = other_peds_with_position[x][3]
                    min_distance[1] = x
                    update = True #indica che e' stato trovato un pedone con distanza minore di quella di default表示发现行人的距离短于默认行人
                x+=1

            # solo se e' stato trovato un pedone con distanza minore di quella di default salviamo questo pedone nell'array sociale
            # 只有当发现行人的距离小于默认距离时，我们才会将此行人保存在社交数组中
            if(update == True):
                # salviamo nell'array alla posizone j-esima le coordinate del pedone con distanza minore
                # 将阵列中距离最短的行人坐标保存到第j个位置
                my_array[pedindex][j] = other_peds_with_position[min_distance[1]][1] # x del pedone piu' vicino
                my_array[pedindex][j+1] = other_peds_with_position[min_distance[1]][2] # y del pedone piu' vicino
                # e poi eliminiamo dall'array quel pedone cosi' da non ripeterlo
                #然后我们从阵列中消除那个行人，以免重复它
                other_peds_with_position.remove(other_peds_with_position[min_distance[1]])
            j += 2

        #abbiamo ora nell'array sociale del pedone le coordinate dei pedoni vicini in ordine di vicinanza
        #se gli altri pedoni fossero >= grid_size ci potremmo fermare qui
        #la parte che segui copre l'eventualita' che gli altri pedoni siano <= grid_size e quindi
        # che nell'array sociale del pedone attuale ci siano ancora posizioni vuote
        #我们现在按照接近的顺序在行人社交阵列中拥有相邻行人的坐标
		#如果其他棋子> = grid_size，我们可以在此停止
		#所遵循的部分涵盖了其他棋子<= grid_size的可能性
		#在当前行人的社交阵列中仍有空位

        #contiamo quanti pedoni mancheranno da inserire nell'array sociale
        #我们计算在社交阵列中插入多少行人
        num_peds_missing = - num_other_peds + (len(my_array[0])/2)

        # se e' un numero maggiore di 0 riempiremo l'array sociale con le ripetizioni degli altri pedoni
        #如果它是一个大于0的数字，我们将用其他行人的重复填充社交数组
        if(num_peds_missing > 0):
            i = 0

            #per ogni spazio vuoto ripeto i pedoni gia' presenti nell'array a partire dal primo
            #对于每个空的空间，我重复从第一个空间开始已经存在于阵列中的行人
            while i < num_peds_missing:
                my_array[pedindex][int((len(my_array[0])/2 - num_peds_missing + i) *2)] = my_array[pedindex][i*2]
                my_array[pedindex][ int((len(my_array[0])/2 - num_peds_missing + i) *2 + 1)] = my_array[pedindex][i*2+1]
                i+=1

    #stampare l'array sociale
    #打印社交阵列
    i = 0
    while i < len(my_array):
        #if(frame[i,0] != 0):
            #print("pedestrian in frame n " + str(i) + " proximity array : " +str(my_array[i]))
        i+=1

    #print("Frame : " + str(frame))

    return my_array


def getSequenceGridMask(sequence, dimensions, neighborhood_size, grid_size):
    '''
    params:
    参数：
    sequence : array con tutti i frame della sequenza, ha dimensioni SL x MNP x 3
    序列：具有序列的所有帧的数组，具有尺寸SL x MNP x 3
    dimensions : inutile, tenuto per semplificare l'implementaizione e modificare solamente questo file
    维度：无用，需要简化实现并仅修改此文件
    neighborhood_size : inutile, tenuto per semplificare l'implementaizione e modificare solamente questo file
    邻居大小：无用，需要简化实现并仅修改此文件
    grid_size : Quanti pedoni saranno presenti nell'array
    阵列中将有多少行人
    '''
    sl = sequence.shape[0] # estrarre e salvare il parametro sequence_length 提取并保存sequence_length参数
    mnp = sequence.shape[1] #estrarre e salvare il parametro MaxNumPeds 提取并保存MaxNumPeds参数

    #l'array contentente l'array sociale di tutti i pedoni per ogni frame dela sequenza
    #包含序列的每个帧的所有行人的社交数组的数组
    #ha dimensioni sequece_length X MaxNumPeds X grid_size*2
    #尺寸为sequece_length X MaxNumPeds X grid_size * 2
    sequence_mask = np.zeros((sl, mnp, grid_size*2))

    #per ogni frame della sequenza richiama la funzione getGridMark aggiungendo il risultato a sequence_mask
    #对于序列的每个帧，它调用getGridMark函数将结果添加到sequence_mask
    for i in range(sl):
        sequence_mask[i, :, :] = getGridMask(sequence[i, :, :], dimensions, neighborhood_size, grid_size)

    return sequence_mask
