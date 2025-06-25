import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import os
import codecs

# Cria o diretório de logs se ele não existir
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configuração do logger
logger = logging.getLogger("trading_bot")
logger.setLevel(logging.INFO)

# Formato da mensagem de log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- INÍCIO DA CORREÇÃO ---

# 1. Handler para console com codificação UTF-8
#    Isso evita o 'UnicodeEncodeError' no console do Windows ao usar emojis.
#    Usamos 'codecs' para 'envelopar' a saída padrão (stdout) com a codificação correta.
#    O 'replace' garante que, se algum erro de codificação ainda ocorrer, ele será substituído
#    em vez de quebrar a execução.
try:
    # Tenta reconfigurar a saída padrão para UTF-8 (funciona em Python 3.7+)
    sys.stdout.reconfigure(encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
except (TypeError, AttributeError):
    # Fallback para sistemas mais antigos: usa um 'writer' do 'codecs'
    utf8_stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    console_handler = logging.StreamHandler(utf8_stdout)

console_handler.setFormatter(formatter)


# 2. Handler para arquivo com rotação diária e encoding UTF-8
#    Adicionamos o parâmetro 'encoding="utf-8"' para garantir que o arquivo de log
#    também salve os emojis e caracteres especiais corretamente.
file_handler = TimedRotatingFileHandler(
    os.path.join(log_dir, 'trading_bot.log'),
    when="midnight",
    interval=1,
    backupCount=7,
    encoding='utf-8' # <--- Adicionado aqui
)
file_handler.setFormatter(formatter)

# --- FIM DA CORREÇÃO ---

# Adiciona os handlers ao logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)