import pandas as pd
from mnist import MNIST

mndata = MNIST('path')

mndata.select_emnist('balanced')
images, labels = mndata.load_training()

#filter out lowercase letters
bad_labels = [i for i,d in enumerate(labels) if d in [36,37,38,39,40,41,42,43,44,45,46]]
good_labels = [i for i,d in enumerate(labels) if i not in bad_labels]

images_no_lower = [images[i] for i in good_labels]
labels_no_lower = [labels[i] for i in good_labels]

#do truncated svd on the images
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
svd.fit(images_no_lower)
#
#images_svd = svd.transform(images_no_lower)
##
###from sklearn.manifold import TSNE
###X_embedded = TSNE(n_components=2,verbose=99).fit_transform(images_svd)
###df = pd.DataFrame(X_embedded)
#
#df = pd.DataFrame(images_svd)
#df['label'] = labels_no_lower
#df['label2'] = df['label'].map({#0: '0'#,1: '1'#,2: '2'#,3: '3'#,4: '4'#,5: '5'#,6: '6'#,7: '7'#,8: '8'#,9: '9'
#,10: 'A'#,11: 'B'#,12: 'C'#,13: 'D'#,14: 'E'#,15: 'F'#,16: 'G'#,17: 'H'#,18: 'I'#,19: 'J'#,20: 'K'#,21: 'L'
#,22: 'M'#,23: 'N'#,24: 'O'#,25: 'P'#,26: 'Q'#,27: 'R'#,28: 'S'#,29: 'T'#,30: 'U'#,31: 'V'#,32: 'W'#,33: 'X'#,34: 'Y'#,35: 'Z'#}) #
#df_means = df.groupby('label2').mean()
#df_means.to_csv('means.csv')

#visualize in R..

#compuate similarity between characters
def generate_dist_matrix():
    dist_matrix = pd.read_csv('dist_matrix.csv')
    
    dist_matrix.index = dist_matrix.index.map({0: '0',1: '1',2: '2',3: '3',4: '4',5: '5',6: '6',7: '7',8: '8',9: '9',10: 'A'
        ,11: 'B',12: 'C',13: 'D',14: 'E',15: 'F',16: 'G',17: 'H',18: 'I',19: 'J',20: 'K',21: 'L',22: 'M',23: 'N',24: 'O',25: 'P'
        ,26: 'Q',27: 'R',28: 'S',29: 'T',30: 'U',31: 'V',32: 'W',33: 'X',34: 'Y',35: 'Z'})
    dist_matrix.columns = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J'
    ,'K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    return dist_matrix