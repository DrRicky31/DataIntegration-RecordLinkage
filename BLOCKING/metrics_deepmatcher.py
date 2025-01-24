import nltk
nltk.download('punkt_tab')  # Scarica il pacchetto mancante

import deepmatcher as dm

# Carica i dati con la corretta configurazione delle colonne
train, validation, test = dm.data.process(
    path='data',
    train='train.csv',
    validation='validation.csv',
    test='test.csv',
    id_attr='id',  # Colonna identificativa
    left_prefix='left_',  # Prefisso per le colonne della tabella sinistra
    right_prefix='right_',  # Prefisso per le colonne della tabella destra
    label_attr='label',  # Colonna della classe target
    cache='processed_data'  # Abilita il caching su disco per ridurre il carico in RAM
)

# Creazione e addestramento del modello
model = dm.MatchingModel(attr_summarizer='hybrid')  # Usa l'architettura "ibrida"

# Addestramento con batch size ridotto
model.run_train(
    train,
    validation,
    epochs=5,  # Riduci il numero di epoche
    batch_size=8,  # Batch size ottimizzato per l'utilizzo di RAM
    best_save_path='best_model.pth'  # Salva il modello migliore
)

# Valutazione del modello
results = model.run_eval(test, batch_size=8)  # Valutazione con batch ridotto

# Stampa dei risultati completi per esaminare il formato
print("Risultati di valutazione:", results)

# Stampa delle metriche (assumendo che i risultati siano in formato dizionario)
# In caso di tensore, estrai il valore con .item()
print("\nMetriche DeepMatcher:")
print(f"F1 Score: {results.item():.2f}")  # Se il risultato è un tensore
