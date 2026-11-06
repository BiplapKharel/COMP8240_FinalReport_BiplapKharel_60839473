# Emoji labels vs reading (emoji hidden)

input: data\tweets\label_check_200.csv
hand-labelled tweets: 200
of these, tweets with an emoji label: 200
  reader found no clear feeling about Christmas: 105 (52%)
  reader gave a positive/negative label: 95
    agreement with emoji label: 90.5%
    Cohen's kappa: 0.790
    confusion (rows = emoji label, cols = reading):
              neg   pos
      neg      28     6
      pos       3    58

disagreements:
  emoji=1 reading=0  Honestly lately people just want to argue, its nearly Christmas time give it a rest, we should be getting together not f
  emoji=0 reading=1  christmas markets look amazin, I wanna go
  emoji=1 reading=0  Nothing brings holiday cheer more than having to stress about if your car is still going to be here after Christmas
  emoji=0 reading=1  I actually liked most of the "tacky" Christmas sweaters
  emoji=1 reading=0  How do I always spend so much money (that I don't have) on Christmas presents
  emoji=0 reading=1  I don't know if I can hold in what I got jake for Christmas 17 days is a long time
  emoji=0 reading=1  I get way too carried away buying Christmas presents for people, just wanna spoil everyone
  emoji=0 reading=1  OMG Trey Songz's version of All I Want For Christmas YASSSSSS CMON TRIGGA
  emoji=0 reading=1  This Christmas will be a very special Christmas for meeeee
