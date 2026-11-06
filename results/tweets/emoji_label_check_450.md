# Emoji labels vs reading (emoji hidden)

input: data\tweets\hand_labels_450.csv
hand-labelled tweets: 450
of these, tweets with an emoji label: 45
  reader found no clear feeling about Christmas: 23 (51%)
  reader gave a positive/negative label: 22
    agreement with emoji label: 90.9%
    Cohen's kappa: 0.812
    confusion (rows = emoji label, cols = reading):
              neg   pos
      neg       8     1
      pos       1    12

disagreements:
  emoji=0 reading=1  I get way too carried away buying Christmas presents for people, just wanna spoil everyone 😭😭😭🎁
  emoji=1 reading=0  Christmas shopping is breaking my bank :))
