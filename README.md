This is a Heroku-ready version of the Radio Galaxy GAN web app

ON HOST:

`$ docker run -it -p 8502:8501 -p 8891:8888 jupyter/tensorflow-notebook:streamlit`

`$ docker exec -it <image_name> bash`

CONTAINER: 

`$ streamlit run GalaxyGAN.py`
