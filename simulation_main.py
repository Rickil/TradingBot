from Bot import Bot
from config import base_url, userId, password

# Initialize bot
bot = Bot(name="Bob", url=base_url, userId=userId, password=password, simulation=True, simulation_data_file="samples/forex_H1.json")

# Run the simulation
bot.run()