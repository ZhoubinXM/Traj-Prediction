
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_trajectories(true_trajs, pred_trajs, obs_length, name):

    traj_length, maxNumPeds, _ = true_trajs.shape

    # Initialize figure
    plt.figure()


    im = plt.imread('plot/plot.png')
    implot = plt.imshow(im)
    width = im.shape[0]
    height = im.shape[1]

    traj_data = {}
    # 对轨迹中的每一帧
    for i in range(traj_length):
        pred_pos = pred_trajs[i, :]
        true_pos = true_trajs[i, :]

        # 对每辆车
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Not a ped
                continue
            elif pred_pos[j, 0] == 0:
                # Not a ped
                continue
            else:
                # If he is a ped
                if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                    continue

                if (j not in traj_data) and i < obs_length:
                    traj_data[j] = [[], []]

                if j in traj_data:
                    traj_data[j][0].append(true_pos[j, 1:3])
                    traj_data[j][1].append(pred_pos[j, 1:3])

    for j in traj_data:

        c = np.random.rand(3, 1)
        true_traj_ped = traj_data[j][0] 
        pred_traj_ped = traj_data[j][1]
        
        #(p[0]+1)/2为了把数据居中显示
        true_x = [(p[0]+1)/2 for p in true_traj_ped]
        true_y = [(p[1]+1)/2 for p in true_traj_ped]
        pred_x = [(p[0]+1)/2 for p in pred_traj_ped]
        pred_y = [(p[1]+1)/2 for p in pred_traj_ped]
        
        plt.plot(true_x, true_y, color='g', linestyle='solid', marker='o',linewidth=3)
        plt.plot(pred_x, pred_y, color='b', linestyle='solid', marker='x',linewidth=3)
        plt.ylim((0.5, 0.8))#为了放大某一区域的轨迹
        plt.xlim((0.6, 0.9))
        #plt.ylim((0, 1))#全局轨迹
        #plt.xlim((0, 1))

    plt.show()
    #plt.savefig('plot/plot.png') # was 
    plt.gcf().clear()
    plt.close()
def main():
    '''
    Main function
    '''
    f = open('save/2/social_results.pkl', 'rb') #was f = open('save/social_results.pkl', 'rb')
    results = pickle.load(f)

    for i in range(len(results)):
        print (i)
        name = 'sequence' + str(i)
    
        plot_trajectories(results[i][0], results[i][1], results[i][2], name)



if __name__ == '__main__':
    main()
