# run.py
import subprocess
import os
import sys
import shutil

# --- Configuração do Projeto ---
DOCKER_IMAGE_NAME = "gcsbot"
KAGGLE_DATA_FILE = os.path.join("data", "kaggle_btc_1m_bootstrap.csv")
ENV_FILE = ".env"
ENV_EXAMPLE_FILE = ".env.example"
# -----------------------------

def print_color(text, color="green"):
    """Imprime texto colorido no terminal."""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "end": "\033[0m",
    }
    print(f"{colors.get(color, colors['green'])}{text}{colors['end']}")

def run_command(command, shell=True):
    """Executa um comando no shell e para o script se houver erro."""
    print_color(f"\n> Executando: {command}", "blue")
    result = subprocess.run(command, shell=shell)
    if result.returncode != 0:
        print_color(f"Erro ao executar o comando: {command}", "red")
        sys.exit(1)

def check_docker_running():
    """Verifica se o Docker Desktop está em execução."""
    print_color("Verificando se o Docker está em execução...", "yellow")
    try:
        subprocess.run("docker info", shell=True, check=True, capture_output=True)
        print_color("Docker está ativo e pronto.", "green")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_color("ERRO: Docker Desktop não parece estar em execução.", "red")
        print_color("Por favor, inicie o Docker Desktop e tente novamente.", "red")
        sys.exit(1)

def check_env_file():
    """Verifica se o arquivo .env existe e o cria a partir do exemplo se necessário."""
    print_color("Verificando arquivo de configuração .env...", "yellow")
    if not os.path.exists(ENV_FILE):
        print_color("Arquivo .env não encontrado.", "red")
        if os.path.exists(ENV_EXAMPLE_FILE):
            print_color(f"Copiando de {ENV_EXAMPLE_FILE} para {ENV_FILE}...", "yellow")
            shutil.copy(ENV_EXAMPLE_FILE, ENV_FILE)
            print_color(f"SUCESSO: Arquivo {ENV_FILE} criado.", "green")
            print_color("IMPORTANTE: Abra o arquivo .env e preencha suas chaves de API antes de continuar.", "red")
            sys.exit(1)
        else:
            print_color(f"ERRO: {ENV_EXAMPLE_FILE} também não encontrado. Não é possível continuar.", "red")
            sys.exit(1)
    print_color("Arquivo .env encontrado.", "green")

def check_data_files():
    """Verifica se os arquivos de dados essenciais existem."""
    print_color("Verificando arquivos de dados necessários...", "yellow")
    if not os.path.exists(KAGGLE_DATA_FILE):
        print_color(f"ERRO: Arquivo de dados do Kaggle não encontrado em: {KAGGLE_DATA_FILE}", "red")
        print_color("Por favor, baixe o arquivo de dados e coloque-o na pasta 'data' para continuar.", "red")
        sys.exit(1)
    print_color("Arquivo de dados do Kaggle encontrado.", "green")

def initial_setup():
    """Executa todas as verificações de inicialização."""
    print_color("--- Iniciando Setup e Verificação do Ambiente ---", "blue")
    check_env_file()
    check_data_files()
    run_command(f"{sys.executable} -m pip install -r requirements.txt")
    print_color("--- Setup Concluído com Sucesso ---", "green")

def docker_build():
    """Constrói a imagem Docker para o bot."""
    check_docker_running()
    print_color(f"--- Construindo Imagem Docker: {DOCKER_IMAGE_NAME} ---", "blue")
    run_command(f"docker build -t {DOCKER_IMAGE_NAME} .")
    print_color("--- Imagem Docker Construída com Sucesso ---", "green")

def start_bot(mode):
    """Inicia o bot usando Docker no modo especificado."""
    check_docker_running()
    container_name = f"gcsbot-{mode}"
    print_color(f"--- Iniciando Bot em Modo '{mode.upper()}' no container '{container_name}' ---", "blue")

    # Garante que as pastas de dados e logs existam no host
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Mapeia as pastas locais para dentro do container para persistência de dados
    data_volume = f"-v \"{os.getcwd()}/data:/app/data\""
    logs_volume = f"-v \"{os.getcwd()}/logs:/app/logs\""
    
    # Parâmetros Docker
    run_params = "-d --restart always" if mode in ['test', 'trade'] else "--rm"
    
    # Comando final para rodar o Docker
    command = (
        f"docker run {run_params} --name {container_name} "
        f"--env-file .env " # Passa as variáveis de ambiente para o container
        f"-e MODE={mode} " # Define o modo de operação
        f"{data_volume} {logs_volume} {DOCKER_IMAGE_NAME}"
    )
    
    run_command(command)
    
    if mode in ['test', 'trade']:
        print_color(f"Bot iniciado em segundo plano. Para ver os logs, use:", "green")
        print_color(f"python run.py logs", "blue")
    else: # Modo optimize
        print_color(f"Otimização iniciada. O processo pode levar horas ou dias.", "yellow")
        print_color(f"Você pode acompanhar o progresso nesta janela. Para parar, pressione CTRL+C.", "yellow")

def stop_bot():
    """Para e remove todos os containers do bot em execução."""
    check_docker_running()
    print_color("--- Parando e Removendo Containers do Bot ---", "yellow")
    # Este comando para e remove containers cujo nome começa com 'gcsbot-'
    run_command("docker ps -a --filter \"name=gcsbot-\" --format \"{{.ID}}\" | xargs -r docker stop | xargs -r docker rm")
    print_color("Containers parados e removidos com sucesso.", "green")
    
def show_logs():
    """Mostra os logs em tempo real de um container em execução."""
    container_name = "gcsbot-live" # Nome padrão para test/trade, ajuste se necessário
    print_color(f"--- Mostrando Logs do Container '{container_name}' (Pressione CTRL+C para sair) ---", "yellow")
    # Tentamos encontrar o container de trade ou test
    try:
        subprocess.run(f"docker logs -f gcsbot-trade", shell=True)
    except KeyboardInterrupt:
        pass
    except Exception:
        try:
            subprocess.run(f"docker logs -f gcsbot-test", shell=True)
        except KeyboardInterrupt:
            pass
        except Exception:
             print_color(f"Nenhum container de trade/test ativo encontrado.", "red")


def main():
    if len(sys.argv) < 2:
        print_color("Uso: python run.py [comando]", "red")
        print("Comandos disponíveis:")
        print("  setup         - Instala dependências e verifica o ambiente.")
        print("  build         - Constrói a imagem Docker do bot.")
        print("  optimize      - Roda o processo de otimização via Docker.")
        print("  test          - Roda o bot em modo TEST na Testnet 24/7 via Docker.")
        print("  trade         - Roda o bot em modo TRADE na conta real 24/7 via Docker.")
        print("  stop          - Para e remove todos os containers do bot.")
        print("  logs          - Mostra os logs do bot que está rodando em modo test/trade.")
        return

    command = sys.argv[1]

    if command == "setup":
        initial_setup()
    elif command == "build":
        docker_build()
    elif command == "optimize":
        start_bot("optimize")
    elif command == "test":
        start_bot("test")
    elif command == "trade":
        start_bot("trade")
    elif command == "stop":
        stop_bot()
    elif command == "logs":
        show_logs()
    else:
        print_color(f"Comando '{command}' não reconhecido.", "red")

if __name__ == "__main__":
    main()