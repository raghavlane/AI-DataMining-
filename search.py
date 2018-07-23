import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import random
from deap import base
from deap import creator
from deap import tools
from itertools import groupby
import matplotlib.pyplot as plt


def trainTestData(df,pd,features):
	train, test = df[df['is_train']==True], df[df['is_train']==False]
	target = pd.factorize(train['result'])[0]
	clf = RandomForestClassifier(n_jobs=2, random_state=0)
	clf.fit(train[features], target)
	prediction = clf.predict(test[features])
	expected = pd.factorize(test['result'])[0]
	TP=0
	TN=0
	FP=0
	FN=0
	for i in range(0,len(prediction)):
		if expected[i]>0:#expected is attack
			if prediction[i]==0:#predection is not attack
				FN = FN +1
			else:#predection is also attack
				TP = TP +1
		else: #expected is not attack
			if prediction[i]==0:#predection is also not attack
				TN = TN +1
			else:#predection is attack
				FP = FP +1
	recall=TP/(TP+FN)
	precision=TP/(TP+FP)
	accuracy=(TP+TN)/len(test)
	return accuracy,precision,recall
	

def evaluateIndividual(df,features,ind):
	feat=[]
	for i in range (0,len(features)-1):
		if(ind[i] =='1'):
			feat.append(features[i])
	features=feat
	X=df[features]
	y=df['result']
	accuracy_list=[]
	precision_list=[]
	recall_list=[]
	skf = StratifiedKFold(n_splits=2)
	for train_index, test_index in skf.split(X, y):
		df.loc[train_index.tolist(),'is_train'] = True
		df.loc[test_index.tolist(),'is_train'] = False
		accuracy,recall,precision=trainTestData(df,pd,features)
		accuracy_list.append(accuracy)
		precision_list.append(precision)
		recall_list.append(recall)
	fin_accuracy= sum(accuracy_list)/len(accuracy_list)
	fin_precision = sum(precision_list)/len(precision_list)
	fin_recall = sum(recall_list)/len(recall_list)
	fin_fitness = (0.6* fin_accuracy )+(0.2*fin_precision)+(0.2*fin_recall)
	print ("fitness= " + str(fin_fitness))
	return fin_fitness
globaldataframe= pd.DataFrame({'A' : []})
def readData():
	global globaldataframe
	if globaldataframe.empty:
		
		##Constants used for mapping to numbers
		protocol=['icmp','tcp','udp']
		service=['http','smtp','finger','domain_u','auth','telnet','ftp','eco_i','ntp_u','ecr_i','other','private','pop_3','ftp_data','rje','time','mtp','link','remote_job','gopher','ssh','name','whois','domain','login','imap4','daytime','ctf','nntp','shell','IRC','nnsp','http_443','exec','printer','efs','courier','uucp','klogin','kshell','echo','discard','systat','supdup','iso_tsap','hostnames','csnet_ns','pop_2','sunrpc','uucp_path','netbios_ns','netbios_ssn','netbios_dgm','sql_net','vmnet','bgp','Z39_50','ldap','netstat','urh_i','X11','urp_i','pm_dump','tftp_u','tim_i','red_i']
		flag=['flag','SF','S1','REJ','S2','S0','S3','RSTO','RSTR','RSTOS0','OTH','SH']
		result=['normal.','buffer_overflow.','loadmodule.','perl.','neptune.','smurf.','guess_passwd.','pod.','teardrop.','portsweep.','ipsweep.','land.','ftp_write.','back.','imap.','satan.','phf.','nmap.','multihop.','warezmaster.','warezclient.','spy.','rootkit.']

		#opening the features list and reading 
		file =open('C:/Users/ragha/OneDrive/Desktop/ECE569A_Assignment1_V00890735/features.txt','r')
		col=file.read().split('\n')
		file.close()
		data=[]


		#opening the dataset, mapping the data to values and adding to data frame
		filepath = 'C:/Users/ragha/OneDrive/Desktop/ECE569A_Assignment1_V00890735/kddcup.data_10_percent_corrected'
		with open(filepath) as fp:
			line = fp.readline()
			while line:
				data_array=line.split('\n')[0].split(',')
				data_array[1]=protocol.index(data_array[1])
				data_array[2]=service.index(data_array[2])
				data_array[3]=flag.index(data_array[3])
				data.append(data_array)
				line = fp.readline()
		globaldataframe= pd.DataFrame(data, columns=col)
	return globaldataframe
	
def main():
		
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMax)

	toolbox = base.Toolbox()
	# Attribute generator 
	toolbox.register("attr_bool", random.randint, 0, 1)
	# Structure initializers
	toolbox.register("individual", tools.initRepeat, creator.Individual, 
		toolbox.attr_bool, 41)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	def evalOneMax(individual):
		print(individual)
		df=readData()
		features = df.columns[:41]
		return evaluateIndividual(df,features,"".join(str(individual))),


	toolbox.register("evaluate", evalOneMax)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
	toolbox.register("select", tools.selTournament, tournsize=3)

	pop = toolbox.population(n=10)

	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	# CXPB  is the probability with which two individualsare crossed
	# MUTPB is the probability for mutating an individual
	CXPB, MUTPB = 0.5, 0.2

	fits = [ind.fitness.values[0] for ind in pop]

	# Variable keeping track of the number of generations
	g = 0
	# Begin the evolution
	maxfitness= []
	meanfitness = []
	while max(fits) < 0.95 and g < 100:
		# A new generation
		g = g + 1
		print("-- Generation %i --" % g)

		# Select the next generation individuals
		offspring = toolbox.select(pop, len(pop))
		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))
		# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:
			if random.random() < MUTPB:
				toolbox.mutate(mutant)
				del mutant.fitness.values
				
		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		pop[:] = offspring
		# Gather all the fitnesses in one list and print the stats
		fits = [ind.fitness.values[0] for ind in pop]
		
		length = len(pop)
		mean = sum(fits) / length
		sum2 = sum(x*x for x in fits)
		std = abs(sum2 / length - mean**2)**0.5
		maxfitness.append(max(fits))
		meanfitness.append(mean)
		print("  Min %s" % min(fits))
		print("  Max %s" % max(fits))
		print("  Avg %s" % mean)
		print("  Std %s" % std)
		grouped_maxfit = [(k, sum(1 for i in g)) for k,g in groupby(maxfitness)]
		last_group = grouped_maxfit[len(grouped_maxfit)-1]
		if ((last_group[0] == max(fits) )and (last_group[1]>=3)):
			break
	plt.plot(meanfitness)
	plt.ylabel('mean fitness')
	plt.show()
	plt.plot(maxfitness)
	plt.ylabel('max fitness')
	plt.show()

main()
#df=readData()
#features = df.columns[:41]
#evaluateIndividual(df,features,"11111111111111111111111111111111111111111"),