# Dzialanie programu

Program porownuje zdania (prompty z kategoriami), zwraca najbardziej
prawdopodobne przypisanie i entropie zdania (do wykrycia danych-smieci).

### Struktura plikow

- w `data/categories.json` znajduje sie lista kategorii, 
- w `data/prompts.json` znajduje sie lista odpowiedzi do klasyfikacji, 
- w `config.yml` znajduje sie konfiguracja modelu/promptowania
- folder `logs/` posiada logi w formacie DD.MM.RRRR-hh:mm:ss.log
##### **WAZNE**
w folderze `models` automatycznie symlinkowane sa
modele z folderu `$HOME/Modele/` 

### Oprogramowanie

Do zsynchronizowania srodowiska wirtualnego mozna uzyc `uv sync`.

##### Makefile

Logi mozna obejrzec za pomoca komendy `make log`.
`make` wlacza program i zapisuje output do odpowiednio nazwanego pliku `.log`.

