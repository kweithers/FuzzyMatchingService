import pandas as pd
import numpy as np
from tqdm import tqdm

all_ids = pd.read_csv('a.txt')
unmatched = pd.concat( pd.read_excel('b.xlsx',sheet_name=None),ignore_index=True)

import Levenshtein
import distance_matrix_generation as dg

unmatched_ids = unmatched['Unmatched BBID']

dist_matrix = dg.generate_dist_matrix()
possible_matches = []
best_matches = []
n = []
unmatched_ids_actual = []
for j in tqdm(range(len(unmatched_ids))):
    
    i = unmatched_ids[j]
    
    dists = [Levenshtein.distance(i,x) for x in all_ids.PRTCPNT_BONUS_EMP_ID]
    
    a=1
    while True:      
        best = [i for i,d in enumerate(dists) if d==a]
        if best != []:
            break
        else:
            a+=1
    
    if a > 2 or len(i) < 6:
        print('skipped' + i)
        possible_matches.append('????')
        best_matches.append('????')
        n.append('????')
        
        
    else:
        candidates = all_ids['PRTCPNT_BONUS_EMP_ID'][best].values
        distances = np.zeros(len(candidates))
        for c in range(len(candidates)):
            diff_locations = [int(i) for i in range(len(candidates[c])) if candidates[c][i] != unmatched_ids[j][i]]
            #print(diff_locations)
            for d in diff_locations:
                distances[c] += dist_matrix.loc[ candidates[c][d] , unmatched_ids[j][d] ] 
    
        best_matches.append(candidates[np.argmin(distances)])
        
        unmatched_ids_actual.append(i)
        possible_matches.append(all_ids['PRTCPNT_BONUS_EMP_ID'][best].values)
        n.append(a)
    

final = pd.DataFrame({'unmatched':unmatched_ids
                      ,'matches':possible_matches
                      ,'best_match':best_matches
                      ,'edit_distance':n})
final.to_csv('final.csv',index=False)