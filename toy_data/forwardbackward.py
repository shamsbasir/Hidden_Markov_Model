import sys
import numpy as np

class hmm:
	def __init__(self,index_to_word,index_to_tag):
		self.pr     = []
		self.a      = []
		self.b      = []
		
		self.predicted = []
		self.obs   = []
		self.state = []

	
		self.tag_map   = self.index_map(index_to_tag)
		self.word_map = self.index_map(index_to_word)

		self.indexToTag = self.wordOfmyIndex(index_to_tag)
		self.indexToWord = self.wordOfmyIndex(index_to_word)
		self.LL = 0.0	

		return None

	def parse_hmm_params(self,hmmprior,hmmemit,hmmtrans):
		loc_emit  = []
		loc_trans = []
		# parse prior
		with open(hmmprior, "r") as f:
			self.pr = np.array([float(item.replace("\n", "")) for item in f.readlines()])

		# parse emit 
		with open(hmmemit, "r") as f:
			lines = [line.replace("\n", "") for line in f.readlines()]
			for line in lines:
				loc_emit.append(np.array([float(num) for num in line.split(" ")]))
		self.b = np.array(loc_emit)

		# parse trans
		with open(hmmtrans, "r") as f:
			lines = [line.replace("\n", "") for line in f.readlines()]
			for line in lines:
				loc_trans.append(np.array([float(num) for num in line.split(" ")]))
		self.a = np.array(loc_trans)
		return None

	def index_map(self,index_to_word):
		bag_of_words = np.loadtxt(index_to_word,dtype=str,delimiter="\n",skiprows=0)
		word_indexed = {}
		count = 0
		for word in bag_of_words:
			word_indexed[word]= count
			count +=1
		return word_indexed

	def wordOfmyIndex(self, index_to_word):
		bag_of_words = np.loadtxt(index_to_word,dtype=str,delimiter="\n",skiprows=0)
		index_word  ={}
		count = 0
		for word in bag_of_words:
			index_word[count]=word
			count +=1
		return index_word

	
	
	def writeToConsole(self):
		print("prior : \n {}".format(self.prior))
		print("hmmemit :\n {}".format(self.emit))
		return None


	def parse_input_data(self,test_input):
		with open(test_input,"r") as f:
			text = [line.replace("\n","") for line in f.readlines()]
		for line in text:
			line = line.split(" ")
			obs = []
			tag = []
			for words in line:
				word = words.split("_")[0]
				state = words.split("_")[1]
				tag.append(self.tag_map[state])
				obs.append(self.word_map[word])
			self.obs.append(obs)
			self.state.append(tag)

		return None


	def forward_back(self):
		pr = np.log(self.pr)
		b  = np.log(self.b)
		a = np.log(self.a)
		count = 0
		for x in self.obs: # grabbed each line of the observed data
			K 	   = len(self.pr)
			T 	   = len(x) 
			alpha 	   = np.zeros((T,K))		
			#for j in range(K):
			alpha[0,:] = pr[:]+ b[:,x[0]]
			
			for k in range(1,T):
				for j in range(K):
					additions = a[:,j]+alpha[k-1]			
					m = np.max(additions)
					additions = additions - m
					alpha[k,j] =b[j,x[k]]+m+np.log(np.sum(np.exp(additions)))
			beta = np.zeros((T,K))
			for k in range(T-2,-1,-1):
				for j in range(K):
					additions = b[:,x[k+1]]+beta[k+1]+a[j,:]
					m = np.max(additions)
					additions = additions -m
					beta[k,j]= m + np.log(np.sum(np.exp(additions)))
		
			# predicted tags probability
			tags = np.add(alpha,beta).T
			# predicted indices
			n = np.argmax(tags,axis = 0)
			self.predicted.append(n)
			# log likelihood
			count +=1
			loc_ll = alpha[-1:]
			m = np.max(loc_ll)
			loc_ll = loc_ll - m
			self.LL += (m + np.log(np.sum(np.exp(loc_ll))))
		self.LL =self.LL/count
		#print(" LL = {}".format(self.LL))
		return None
		

	def forwardAndbackward(self):
		count = 0
		for x in self.obs: # grabbed each line of the observed data
			K 	   = len(self.pr)
			T 	   = len(x)
			alpha 	   = np.zeros((T,K))				
			beta  	   = np.zeros((T,K))		
			alpha[0,:] = np.multiply(self.b[:,x[0]], self.pr)
			for t in range(1,T):
				val = np.dot(self.a.T, alpha[t-1,:].T)
				alpha[t,:] = np.multiply(self.b[:,x[t]],val.T)
			
			beta[-1:] = 1
			for t in range(T-2,-1,-1):
				val = np.multiply(self.b[:,x[t+1]],beta[t+1,:])
				beta[t,:]= np.dot(self.a, val.T)
				beta[t,:]/= np.sum(beta[t,:])	
			
			# predicted tags probability
			tags = np.multiply(alpha,beta).T
			# predicted indices
			n = np.argmax(tags,axis =0)
			self.predicted.append(n)
			# log likelihood
			count +=1
			loc_ll = alpha[-1:]
			self.LL += np.log(np.sum(loc_ll))
		self.LL /=count
		#print(" LL = {}".format(self.LL))
		return None

	def postprocess(self,predicted_file,metric_file):
		#print("Average Log-Likelihood: ",self.LL)
		elements = 0
		count = 0	
		for x, y in zip(self.state,self.predicted):
			for i,j in zip(x,y):
				if i!=j : 
					count +=1
				elements +=1
		accuracy = 1 - count/elements
		predicted  = []
		for words,tags in zip(self.obs,self.predicted):
			loc_predict =[]
			for word_index , tag_index in zip(words,tags):
				tag = self.indexToTag[tag_index] 
				word  = self.indexToWord[word_index]
				#print("{}_{}".format(word,tag))
				loc_predict.append('{}_{}'.format(word,tag))
			predicted.append(loc_predict)
		# time to write them to file
		with open(predicted_file, "w") as f:
			lines = [" ".join(item) for item in predicted]
			f.write("\n".join(lines))
		with open(metric_file, "w") as f:
			f.write("Average Log-Likelihood: {}\n".format(self.LL))
			f.write("Accuracy: {}".format(accuracy))
		return None


def main():
	args 	      = sys.argv
	test_input    = args[1]
	index_to_word = args[2]
	index_to_tag  = args[3]
	hmmprior      = args[4]
	hmmemit	      = args[5]
	hmmtrans      = args[6]
	predicted_file= args[7]
	metric_file   = args[8]
	
	fb = hmm(index_to_word,index_to_tag)
	fb.parse_hmm_params(hmmprior,hmmemit,hmmtrans) 
	fb.parse_input_data(test_input)
	#fb.forwardAndbackward()
	fb.forward_back()	
	fb.postprocess(predicted_file,metric_file) 
	
if __name__ == "__main__":
	main()
