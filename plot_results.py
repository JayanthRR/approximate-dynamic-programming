from matplotlib import pyplot as plt
from rollout import Result, SubRes
import pickle

with open('reference.pkl','rb') as file:
    reference = pickle.load(file)

with open('result.pkl', 'rb') as file:
    result = pickle.load(file)

print(reference)

fig1, axarr1 = plt.subplots(2, 2)

for i in range(4):
    # if i%2 == 0:
    axarr1[i//2, i%2].semilogy(reference['heights'], result.dp.time[i,:], label='dp', color='red')
    axarr1[i//2, i%2].semilogy(reference['heights'], result.rollout.time[i,:], label='rollout', color='green')
    axarr1[i//2, i%2].semilogy(reference['heights'], result.greedy.time[i,:], label='greedy', color='blue')
    axarr1[i//2, i%2].set_title('prob : %.2f'%(reference['probs'][i]))
    axarr1[i//2, i%2].grid()

    axarr1[i // 2, i % 2].set_xlabel('height of tree')
    axarr1[i // 2, i % 2].set_ylabel('time taken (s)')

axarr1[0,0].legend()
# fig1.suptitle('Time taken vs height of tree for various probabilities')
fig1.tight_layout()
fig1.savefig('time.svg', dpi=fig1.dpi)


fig2, axarr2 = plt.subplots(2, 2)

for i in range(4):
    # if i%2 == 0:
    axarr2[i//2, i%2].plot(reference['heights'], result.dp.acc[i,:], label='dp', color='red')
    axarr2[i//2, i%2].plot(reference['heights'], result.rollout.acc[i,:], label='rollout', color='green')
    axarr2[i//2, i%2].plot(reference['heights'], result.greedy.acc[i,:], label='greedy', color='blue')
    axarr2[i//2, i%2].set_title('prob : %.2f'%(reference['probs'][i]))
    axarr2[i//2, i%2].grid()

    axarr2[i // 2, i % 2].set_xlabel('height of tree')
    axarr2[i // 2, i % 2].set_ylabel('accuracy')


axarr2[0,0].legend()
# fig2.suptitle('Accuracy vs height of tree for various probabilities')
fig2.tight_layout()
fig2.savefig('accuracy.svg', dpi=fig2.dpi)

plt.show()


