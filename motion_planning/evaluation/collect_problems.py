import os
import numpy as np
import pickle

robot = 'magician'

dir_path = "../../models/motion_planning/problems/"
all_file = os.listdir(dir_path + 'pkl/')
robot_file = [file for file in all_file if file.startswith(robot) and "supersafe" in file]

results = []
for file in robot_file:
	with open(dir_path + 'pkl/' + file, 'rb') as handle:
		while True:
			try:
				results.append(pickle.load(handle))
			except:
				break

collected_results = {key: [] for key in results[0].keys()}
i = 0

for result in results:
	if 0.5 < result['time_dict']['total_time'] < 2:
		for key in collected_results.keys():
			collected_results[key].append(result[key])
		i += 1
	if i >= 1000:
		break
print(i)

with open(dir_path + f"refined_pkl/{robot}_hardd.pkl", "wb") as handle:
	pickle.dump(collected_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
