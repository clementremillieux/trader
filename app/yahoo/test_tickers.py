from yahooquery import Screener

# Créer une instance du Screener
s = Screener()

# Récupérer les données du screener pour toutes les cryptomonnaies aux États-Unis
data = s.get_screeners("all_cryptocurrencies", count=250)


# Extraire les symboles des cryptomonnaies
symbols = [item["symbol"] for item in data["all_cryptocurrencies_us"]["quotes"]]

# Afficher les symboles
print(symbols)
