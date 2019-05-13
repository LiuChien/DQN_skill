'''
input log.txt path

Plot diagram of score, score_std of each episode
Plot diagram of highest score, average acore of whole past episodes


'''


from argparse import ArgumentParser
import matplotlib.pyplot as plt

# Create parser
parser = ArgumentParser()
parser.add_argument('path', type=str, 
                    help='path to log.txt')
parser.add_argument('--title', type=str,
                    help='title of diagram')
args = parser.parse_args()

# Decode log.txt
def decode_file(path):
    score = []
    score_std = []

    with open(path, 'r') as f:
        lines = f.read().split('\n')
        for l in lines:
            if 'ave_score:' in l:
                score.append(decode_line(l, 'ave_score:'))

            elif 'ave_score_std:' in l:
                score_std.append(decode_line(l, 'ave_score_std:'))

    return score, score_std

def decode_line(l, item):
    '''
    type(l) == str
    type(item) == str
    '''
    l = l.replace(item, '')
    l = l.replace(' ', '')

    return float(l)

def generate_score(score):
    ave_score = []
    max_score = []

    for i in range(len(score)):
        target = score[:i+1]
        ave_score.append(sum(target) / len(target))
        max_score.append(max(target))

    return ave_score, max_score

def plot_diagram(score, score_std, ave_score, max_score, title):
    plt.plot(score, label='score')
    plt.plot(score_std, label='std')
    plt.plot(ave_score, label='avg score')
    plt.plot(max_score, label='max history score')
    plt.legend(loc='lower right')
    plt.title(title)

    plt.show()

    filename = title + ".png"
    plt.savefig(filename)


if __name__ == '__main__':
    score, score_std = decode_file(args.path)
    ave_score, max_score = generate_score(score)

    plot_diagram(score, score_std, ave_score, max_score, args.title)
    

'''
******* Episode: 1 ********
ave_score: 181.0
ave_score_std: 51.856
ave_action_reward: 0.3861669662619688
[0]: 205
[1]: 205
[2]: 215
[3]: 210
[4]: 177
[5]: 237
[6]: 197
[7]: 216
[8]: 194
[9]: 186
[10]: 254
[11]: 239
[12]: 167
[13]: 217
[14]: 219
[15]: 230
[16]: 233
[17]: 196
[9, 10, 11, 5, 5, 10]: 222
[6, 7, 10, 17, 4, 1]: 313
[2, 17, 0, 14, 2, 2]: 222
[1, 5, 10, 3, 14, 5]: 225
***************************
'''