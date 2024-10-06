
The initial values make intuitive sense since since a unigram does not have any sentence context each time it considers the following token. The bigram and trigram predictably improve the LM's perplexity. However, this trend does not continue as the perplexity becomes quite a bit worse as more and more context is introduced. We predict this is because the scope becomes too narrow for the model once the size of history surpasses two.

Of the original 151662 types, 1250 are in the final vocab
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 34028.10it/s]
Average per-line perplexity for 1-gram LM: 146.39

 ------ 

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:02<00:00, 198.17it/s]
Average per-line perplexity for 2-gram LM: 70.55

2-GRAM SAMPLE:
<s> this mess with a very little plot is not to film , since shows many of credits ) .

 ------ 

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 373.55it/s]
Average per-line perplexity for 3-gram LM: 54.01

3-GRAM SAMPLE:
<s> <s> dumb , the french in the form of entertainment ... < br / > around half an hour aside <UNK> of going to be made , well , not many . simply put , she seemed like great works of " alien " and " material girls " is totally what living life is the high quality not generally the men around the turn of events that he would like to see the movie was good ( or worst film director and the story turned into just about all i would recommend either of them are bad in this episode is a good movie that is the true story of nice " high tension " . this picture he says in the story almost the same time . ' this great film but i at least half of the film are the result looks rather worse for bad . only for a fake their death , i watched the video . they make the movie that made me feel . the film is so totally wrong ! < br / > < br / > to me this isn't . the music score by bruce lee himself -

 ------ 

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 763.22it/s]
Average per-line perplexity for 4-gram LM: 253.06

4-GRAM SAMPLE:
<s> <s> <s> the first " alien " and " you are really a huge fan of her husband and another a <UNK> who's half of a plot twist , 1 , 2 , 3 & 4 , " the mysterious island . * <UNK> from * * * who shot you was <UNK> ' the producers ' of the martin family , to not be a great film . when his wife and son are able to look for all fans of the violent scenes were well done . but , robert <UNK> as henry is a time to really take deep <UNK> view of american society of the late ' 90 ' s . the story takes place . < br / > < br / > < br / > < br / > i just wish it were an art <UNK> through a door the other figure <UNK> the story to make up for the kids but if you do not know how many stupid <UNK> in love with a man ( and woman ) . this film really never got on the case and especially with him as she leaves the room to do to

 ------ 

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 1304.51it/s]
Average per-line perplexity for 5-gram LM: 620.53

5-GRAM SAMPLE:
<s> <s> <s> <s> by the early ' <UNK> . it isn't easy to find , if that <UNK> . it is my favorite " cult classic " are <UNK> a lot like real life and what makes this movie so wonderful is how good a job <UNK> and his <UNK> dog < br / > and while we're on the subject , you might find this movie entertaining , i'm <UNK> it takes more than a red <UNK> and the greatest <UNK> of this clearly pathetic movie . i'll admit , i wasn't expecting that . it was bad on so many <UNK> . < br / > < br / > < br / > the direction isn't awful but it's just generally <UNK> . what was good about the first one ? a lot of this performance is funny and one bit at the end that completely <UNK> the very dark ending of the real story ... but as a movie , the people at the top control everything and when she finds out her husband ( <UNK> of in <UNK> back and quite easily at that and a new hero is <UNK> as a horror movie

 ------