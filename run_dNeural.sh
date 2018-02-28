#! /bin/bash



#recreate us
python dneural_style.py --content content/us.png \
	--label us_construct
	--styles styles/picasso.jpg \
	--output out/us.jpg \
	--checkpoint-output out/us%s.jpg \
	--checkpoint-iterations 200 \
	--iterations 2000  \
	--rContent

# DO picasso
python dneural_style.py --content content/us.png \
	--label picasso
	--styles styles/picasso.jpg \
	--initial content/us.png \	
	--output out/picasso.jpg \
	--checkpoint-output out/picasso%s.jpg \
	--checkpoint-iterations 200 \
	--iterations 2000  \
	--preserve-colors	

python dneural_style.py --content content/us.png \
	--label picasso_construct
	--styles styles/picasso.jpg \
	--output out/picasso_construct.jpg \
	--checkpoint-output out/picasso_construct%s.jpg \
	--checkpoint-iterations 200 \
	--iterations 2000  \
	--rStyle



	
# DO SALV
python dneural_style.py --content content/us.png \
	--label salv
	--styles styles/salv.png \
	--output out/salv.jpg \
	--initial content/us.png \	
	--checkpoint-output out/salv%s.jpg \
	--checkpoint-iterations 200 \
	--iterations 2000  \
	--preserve-colors	

python dneural_style.py --content content/us.png \
	--label salv_construct
	--styles styles/salv.png \
	--output out/salv_construct.jpg \
	--checkpoint-output out/salv_construct%s.jpg \
	--checkpoint-iterations 200 \
	--iterations 2000  \
	--rStyle
	
	
	
# DO Anime
python dneural_style.py --content content/us.png \
	--label anime
	--styles styles/anime.jpg \
	--initial content/us.png \
	--output out/anime.jpg \
	--checkpoint-output out/anime%s.jpg \
	--checkpoint-iterations 200 \
	--iterations 2000  \
	--preserve-colors	

python dneural_style.py --content content/us.png \
	--label anime_construct
	--styles styles/anime.jpg \
	--output out/anime_construct.jpg \
	--checkpoint-output out/anime_construct%s.jpg \
	--checkpoint-iterations 200 \
	--iterations 2000  \
	--rStyle	
	
	
	
# DO stary
python dneural_style.py --content content/us.png \
	--label stary
	--styles styles/stary.jpg \
	--initial content/us.png \
	--output out/stary.jpg \
	--checkpoint-output out/stary%s.jpg \
	--checkpoint-iterations 200 \
	--iterations 2000  \
	--preserve-colors	

python dneural_style.py --content content/us.png \
	--label stary_construct
	--styles styles/stary.jpg \
	--output out/stary_construct.jpg \
	--checkpoint-output out/stary_construct%s.jpg \
	--checkpoint-iterations 200 \
	--iterations 2000  \
	--rStyle		
	
	
	
	