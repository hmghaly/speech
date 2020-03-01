import os, re
from scipy import signal
from scipy.io import wavfile

def get_times_phonemes(phn_fpath,wav_fpath):
  sample_rate, samples = wavfile.read(wav_fpath)
  #samples = np.mean(samples, axis=1) #to handle 2-channels
  frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

  fopen=open(phn_fpath)
  file_ph_list=[]
  for line in fopen:
    line_split=re.split("\s",line.strip("\n")) 
    #print(line_split)
    #if len(line_split)<2: continue
    #if line_split[1]!="phn": continue
    #if len(line_split)!=4: continue
    if len(line_split)!=3: continue
    t0_str, t1_str, ph= line_split
    t0=float(t0_str)/16000 #*0.001 - 16khz samples
    t1=float(t1_str)/16000 #*0.001
    file_ph_list.append((ph, t0,t1))
  tmp_final_list=[]
  #print(times)
  for ti in times:
    
    for ph,t0,t1 in file_ph_list:
      #print(ph,t0,t1, ">>>", ti)
      if ti>=t0 and ti<=t1: 
        tmp_final_list.append((ti,ph))
        #print(ti,ph)
        break
  # time_ph_freq=[]
  # spectrogram_transposed=spectrogram.transpose()
  # for t_ph, sp in zip(our_final_list,spectrogram_transposed):
  #   t_,ph=t_ph
  #   time_ph_freq.append((t_,ph,sp))
  return tmp_final_list, spectrogram


phonemes=['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

phn_fpath="speech/timit/ftbr0/sx381.phn.txt"
wav_fpath="speech/timit/ftbr0/sx381.wav"
phn_fpath="speech/timit/ftbr0/sx381.phn"

  
final_list,spec=get_times_phonemes(phn_fpath,wav_fpath)
print(spec.shape)
# for a in our_final_list:
#   print(a)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#setting up the RNN to accept a sequence of freq values at each time step, and predict the corresponding phoneme
torch.manual_seed(1)
random.seed(1)

models_dir="speech/models"
if not os.path.exists(models_dir): os.mkdir(models_dir)

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_depth,number_layers, batch_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.lstm = nn.LSTM(input_size, hidden_size,number_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_size, output_depth)
        self.hidden = self.init_hidden()

    def forward(self, feature_list): #emeds are the list of features for each word in the sentece
        #sent_size=len(embeds)
        lstm_out, _ = self.lstm( feature_list.view(len( feature_list), 1, -1))
        tag_space = self.hidden2out(lstm_out.view(len( feature_list), -1))
        #print(tag_space.view([1,1,1]))
        tag_scores = torch.sigmoid(tag_space)
        #return tag_scores
        return tag_space
       
    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_size),
                torch.zeros(1, self.batch_size, self.hidden_size))  

phonemes=['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

#experiment parameter
exp_name="timit_full_test"

n_epochs=20

n_input=129
n_hidden =128
n_layers=2
n_depth=len(phonemes)

rewrite_info=False

exp_info_fname="%s_info.txt"%exp_name
exp_info_fpath=os.path.join(models_dir,exp_info_fname)
if rewrite_info: 
  info_fopen=open(exp_info_fpath,"w")
  info_fopen.close()

#else: info_fopen=open(exp_info_fpath,"a")




rnn = RNN(n_input, n_hidden, n_depth,n_layers)
loss_func = nn.MSELoss()

LR=0.0001 #let's play with this
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
rnn.hidden = rnn.init_hidden()

timit_dir="speech/timit"
timit_dir="speech/timit_full/test"
timit_train_dir="speech/timit_full/train"
timit_test_dir="speech/timit_full/test"
all_wav_files=[]
for r,d,files in os.walk(timit_dir):
  for f in files:
    #print(f)
    if not f.endswith(".wav"): continue
    wav_fpath=os.path.join(r,f)
    phn_fpath=wav_fpath.replace(".wav",".phn")
    if not os.path.exists(phn_fpath): continue
    all_wav_files.append(wav_fpath)
    #print(wav_fpath)

print(len(all_wav_files), all_wav_files[:10])
train_files=all_wav_files[:-20]
test_files=all_wav_files[-20:]

for epoch in range(n_epochs):
  correct_counter=0.
  total_counter=0.
  print("epoch", epoch)
  all_current_loss=0
  rnn.zero_grad()
  for f_,wav_fpath in enumerate(train_files):
    if f_%20==0: print("file #", f_)
    #phn_fpath=wav_fpath.replace(".wav",".phn.txt")
    phn_fpath=wav_fpath.replace(".wav",".phn")
    #extract features and specify expected outcome
    final_list,spec=get_times_phonemes(phn_fpath,wav_fpath)
    #spec=np.log(spec)
    spec_transpose=torch.tensor(spec.transpose()) 
    if len(final_list)!=len(spec_transpose): continue #not sure why it can be like this, need to investigate
    
    output = rnn(spec_transpose) #network output
    for output_slice,fl in zip(output,final_list):
      cur_preds=[(phonemes[i0],i0,v.item() ) for i0,v in enumerate(list(output_slice))]
      cur_preds.sort(key=lambda x:-x[-1])
      total_counter+=1
      if fl[1]==cur_preds[0][0]: correct_counter+=1
      # if i_>n_epochs-5 and False:
      #   print("correct", fl[1], cur_preds[:3])

    expected=[]
    for fl in final_list: #get the expected tensor to compare with network output
      cur_one_hot=[0.]*len(phonemes)
      cur_one_hot[phonemes.index(fl[1])]=1.
      expected.append(cur_one_hot)

    expected_tensor=torch.tensor(expected)

    loss=loss_func(output,expected_tensor) #get the loss by comparing the output with the expected
    loss.backward()
    optimizer.step()
    all_current_loss+=loss

  model_name="model_L%s_H%s_E%s.model"%(n_layers,n_hidden,epoch)
  model_fpath=os.path.join(models_dir,model_name)
  torch.save(rnn.state_dict(), model_fpath)


  test_total_counter=0.
  test_correct_counter=0.
  test_loss=0.
  with torch.no_grad():
    for f_,wav_fpath in enumerate(test_files):
      #phn_fpath=wav_fpath.replace(".wav",".phn.txt")
      phn_fpath=wav_fpath.replace(".wav",".phn")
      #extract features and specify expected outcome
      final_list,spec=get_times_phonemes(phn_fpath,wav_fpath)
      #spec=np.log(spec)
      spec_transpose=torch.tensor(spec.transpose()) 
      
      output = rnn(spec_transpose) #network output
      for output_slice,fl in zip(output,final_list):
        cur_preds=[(phonemes[i0],i0,v.item() ) for i0,v in enumerate(list(output_slice))]
        cur_preds.sort(key=lambda x:-x[-1])
        test_total_counter+=1
        if fl[1]==cur_preds[0][0]: test_correct_counter+=1

      expected=[]
      for fl in final_list: #get the expected tensor to compare with network output
        cur_one_hot=[0.]*len(phonemes)
        cur_one_hot[phonemes.index(fl[1])]=1.
        expected.append(cur_one_hot)

      expected_tensor=torch.tensor(expected)

      loss=loss_func(output,expected_tensor) #get the loss by comparing the output with the expected
      test_loss+=loss


    test_accuracy=test_correct_counter/test_total_counter


  accuracy=correct_counter/total_counter
  print("epoch", epoch, "loss", round(all_current_loss.item(),4) , "accuracy",round(accuracy,3), "test loss", round(test_loss.item(),4) , "test accuracy",round(test_accuracy,3) )
  info_line="%s\t%s\t%s\t%s\t%s\t%s\n"%(model_name,epoch,round(all_current_loss.item(),4),round(accuracy,4),round(test_loss.item(),4),round(test_accuracy,4))

  info_fopen=open(exp_info_fpath,"a")
  info_fopen.write(info_line)
  info_fopen.close()
  #print("correct_counter", correct_counter, "total_counter", total_counter)
#now predicting on one of TMIT test files using one of the models 
#loading a trained model
rnn = RNN(n_input, n_hidden, n_depth,n_layers)
rnn.load_state_dict(torch.load("speech/models/model_L2_H128_E39.model"))
rnn.eval()
wav_fpath = test_files[0] #"speech/ar_digits_test/2-5.wav"
phn_fpath=wav_fpath.replace(".wav",".phn.txt")
#extract features and specify expected outcome
final_list,spec=get_times_phonemes(phn_fpath,wav_fpath)
spec_transpose=torch.tensor(spec.transpose()) 
output = rnn(spec_transpose) #network output
for output_slice,fl in zip(output,final_list):
  correct_ph=fl[1]
  cur_preds=[(phonemes[i0],round(v.item(),2) ) for i0,v in enumerate(list(output_slice))]
  cur_preds.sort(key=lambda x:-x[-1])
  cur_preds_phon=[v[0] for v in cur_preds]
  where=cur_preds_phon.index(correct_ph)
  if where>2:
    print("correct", fl,"found in", where ,">>> predicted", cur_preds[:3], "val:", cur_preds[where])
  else:
    print("correct", fl,"found in", where ,">>> predicted", cur_preds[:3])

#and also predicting on a file not annotated
wav_fpath = "speech/ar_digits_test/2-5.wav"
sample_rate, samples = wavfile.read(wav_fpath)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
spec_transpose= torch.tensor(spectrogram.transpose())
#print(spec_transpose.shape)
#spec_transpose=spec.transpose()

output = rnn(spec_transpose) #network output
for t_,output_slice in zip(times,output):
  cur_preds=[(phonemes[i0],round(v.item(),2) ) for i0,v in enumerate(list(output_slice))]
  cur_preds.sort(key=lambda x:-x[-1])
  print(round(t_,2), cur_preds[:5])
