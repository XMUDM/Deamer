#python example to infer document vectors from trained doc2vec model
import gensim.models as g
import codecs

category = 'Movies_and_TV'
k_core = '_5'

#parameters
model = category + "//PreData//model_metadata" + k_core + ".bin"
train_docs = category + "//PreData//metadata_description_title" + k_core + ".txt"
output_file1 = category + "//feature_data//metadata_description_title_vectors" + k_core + ".txt"
#test_docs="PreData//Instruments_metadata_description_title.test"
#output_file2="feature_data//Instruments_metadata_description_title_vectors.test"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000
'''
sentence = []
sentence.append('seek a dream')
sentence.append('did have a pad')
test_docs = [ x.strip().split() for x in sentence ]
'''
print('category = ', category, ', start to infer item medadata to vector')
#load model
m = g.Doc2Vec.load(model)

#train_docs = [ x.strip().split() for x in codecs.open(train_docs, "r", "utf-8").readlines() ]
sentence = []
with open(train_docs, "r",encoding = "utf-8") as f:
    for line in f.readlines() :
        sentence.append(line)
train_docs = [ x.strip().split() for x in sentence ]
print(len(train_docs))

#print(train_docs)

#infer test vectors
output1 = open(output_file1, "w")
for d in train_docs:
    #print(m.infer_vector(d, alpha=start_alpha, steps=infer_epoch))
    output1.write( " ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" )
output1.flush()
output1.close()

'''
test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]

#print(test_docs)

#infer test vectors
output2 = open(output_file2, "w")
for d in test_docs:
    #print(m.infer_vector(d, alpha=start_alpha, steps=infer_epoch))
    output2.write( " ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" )
output2.flush()
output2.close()
'''