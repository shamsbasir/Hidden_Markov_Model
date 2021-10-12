import sys
import numpy as np 
# Program name : Hidden Morkov Model
class learnhmm:
	def __init__(self,index_to_word,index_to_tag):
		self.word_map    = self.index_map(index_to_word)
		self.state_map   = self.index_map(index_to_tag)
		self.hstat_count = len(self.state_map)
		self.obs_count   = len(self.word_map)
		self.prior_prob  = np.zeros((self.hstat_count))
		self.trans_prob  = np.zeros((self.hstat_count,self.hstat_count))
		self.emit_prob   = np.zeros((self.hstat_count,self.obs_count))
		return None

	def index_map(self,index_to_word):
		bag_of_words = np.loadtxt(index_to_word,dtype=str,delimiter="\n",skiprows=0)
		word_indexed = {}
		count = 0
		for word in bag_of_words:
			word_indexed[word]= count
			count +=1
		return word_indexed
	
	def train(self,train_data):
		with open(train_data) as f:
			text = [item.replace("\n", "").split(" ") for item in f.readlines()]
			#print(text)	
		#for line in text[:10]:
		for line in text:
			#print("line = {}".format(line))
			for index, item in enumerate(line):
				word, tag = item.split("_")
				word_index, tag_index = int(self.word_map[word]), int(self.state_map[tag])
				#print("word_index = {}, tag_index= {}".format(word_index,tag_index))
				if index == 0:
					self.prior_prob[tag_index] += 1
				if index != len(line) - 1:
					next_item = line[index + 1]
					next_word, next_tag = next_item.split("_")
					next_word_index, next_tag_index = int(self.word_map[next_word]),int(self.state_map[next_tag])
					self.trans_prob[tag_index][next_tag_index] += 1
				self.emit_prob[tag_index][word_index] += 1
		self.emit_prob  +=1
		self.prior_prob +=1
		self.trans_prob +=1
		self.prior_prob /=np.sum(self.prior_prob)
		self.trans_prob /=np.sum(self.trans_prob, axis = 1).reshape(int(self.hstat_count), -1) 
		self.emit_prob  /= np.sum(self.emit_prob, axis = 1).reshape(int(self.hstat_count),-1)
		#print("prior : \n {}".format(self.prior_prob))
		#print("emission : \n {}".format(self.emit_prob))
		#print("transmission : \n {}".format(self.trans_prob))
		return None

	def postprocess(self,hmmprior,hmmemit,hmmtrans):
		self.writeTofile(self.prior_prob,hmmprior)
		self.writeTofile(self.trans_prob,hmmtrans)
		self.writeTofile(self.emit_prob,hmmemit)
	
		return None
	
	def writeTofile(self,matrix,outfile):
		with open(outfile, "w") as f:
			line = []
			for item in matrix:
				try:
					item = ["%.18e"%(digit) for digit in item]
					line.append(" ".join(item))
				except:
					line.append(str(item))
			f.write("\n".join(line))
		return None

def main():
    args = sys.argv
    train_input   = args[1]
    index_to_word = args[2]
    index_to_tag  = args[3]
    hmmprior      = args[4]
    hmmemit       = args[5]
    hmmtrans      = args[6]
    hmm = learnhmm(index_to_word,index_to_tag)
    hmm.train(train_input)
    hmm.postprocess(hmmprior,hmmemit,hmmtrans)
 
if __name__ == "__main__":
    main()
    
