import configparser
import os
def get_config():
    config = configparser.ConfigParser()
    #config.read('../../config.ini')
    config.read('../config.ini')
    return config

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

from datetime import datetime
import telebot

config = get_config()['telegram']
bot = None
try:
  bot = telebot.TeleBot(config['token'])
except:
  pass

def send_message(text: str):
  if not bot is None:
    bot.send_message(chat_id=config['chat_id'], text=text)
  else:
    print(f'MESSAGE: {text}')