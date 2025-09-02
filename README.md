### Panoramica del Sistema

Il sistema implementa un framework avanzato di Federated Learning che permette l'addestramento distribuito di modelli di deep learning su più client mantenendo i dati localmente su ogni dispositivo. Il sistema è progettato per simulare scenari realistici con client eterogenei che presentano diverse capacità computazionali, di rete e di affidabilità.

### Caratteristiche Principali

Il framework si distingue per le seguenti innovazioni:

1. **Simulazione Resource-Aware**: Ogni client virtuale ha caratteristiche di prestazione diverse che influenzano realisticamente il comportamento del training
2. **Politiche di Aggregazione Adaptive**: Implementazione di diverse strategie di selezione e aggregazione basate sulle risorse disponibili
3. **Aggregazione Parziale**: Capacità di procedere con l'aggregazione anche quando non tutti i client rispondono
4. **Confronto Framework**: Sistema di testing automatico che confronta il framework custom con FLOWER
5. **Algoritmi FL Multipli**: Supporto per FedAvg, FedProx, FedYogi e FedAdam

---

## Architettura del Sistema

### Componenti Principali

**Server Federato (federated_server.py)**
Il server federato agisce come coordinatore centrale e gestisce:

- Gestione delle connessioni di più client simultanei
- Mantenimento dello stato delle risorse di ogni client
- Coordinamento dei round di addestramento globali
- Selezione intelligente dei client basata su politiche configurabili
- Aggregazione dei modelli utilizzando algoritmi avanzati
- Monitoraggio e logging dettagliato delle performance

**Client Federato (federated_client.py)**
Ogni client federato mantiene:

- Dataset locale privato con gestione autonoma dei dati
- Capacità di addestramento locale con algoritmi FL specifici
- Simulazione realistica delle proprie risorse computazionali
- Comunicazione asincrona con il server
- Calcolo e reporting delle metriche locali

**Sistema di Risorse (client_resources.py)**
Gestisce la simulazione delle caratteristiche dei client:

- Potenza computazionale variabile (0.5x a 2.0x rispetto al baseline)
- Larghezza di banda simulata (1.0 a 10.0 Mbps)
- Affidabilità del client (probabilità di successo 0.7-1.0)
- Tempi di computazione e trasmissione realistici

### Architettura di Comunicazione

```
[Server FL] ←→ [Client 1] [Client 2] [Client 3] [Client 4]
     ↑
[Aggregatore] + [Policy Selection] + [FL Algorithm]
```

Il sistema utilizza comunicazione bidirezionale asincrona basata su WebSocket, permettendo:

- Connessioni persistenti tra server e client
- Invio asincrono di modelli e metriche
- Gestione robusta di disconnessioni e timeout
- Distribuzione universale dei modelli aggiornati

---

## Simulazione delle Risorse Client

### Classe ClientResources

La simulazione delle risorse è implementata attraverso la classe ClientResources che modella tre aspetti fondamentali:

**Compute Power (Potenza Computazionale)**
Rappresenta la capacità di calcolo del client relativa a un dispositivo baseline:

- 0.5: Dispositivi IoT con hardware molto limitato (training 2x più lento)
- 0.8: Smartphone/Tablet con capacità moderate (training 25% più lento)
- 1.0: Desktop standard utilizzato come baseline
- 1.5: Workstation ad alte prestazioni (training 50% più veloce)
- 2.0: Server/GPU dedicati (training 2x più veloce)

**Bandwidth (Larghezza di Banda)**
Velocità di connessione di rete del client in Megabit per secondo:

- 1.0-2.0 Mbps: Connessioni lente (2G/3G, WiFi debole)
- 3.0-5.0 Mbps: Connessioni moderate (4G, WiFi domestico)
- 6.0-8.0 Mbps: Connessioni veloci (4G+, Fibra standard)
- 9.0-10.0 Mbps: Connessioni ottimali (5G, Fibra ad alta velocità)

**Reliability (Affidabilità)**
Probabilità che il client completi con successo un round di training:

- 0.7-0.8: Client instabili con frequenti disconnessioni
- 0.85-0.9: Client moderatamente affidabili
- 0.95-1.0: Client altamente stabili e affidabili

### Simulazione Realistica

Il sistema introduce variabilità stocastica per simulare condizioni reali:

```
Tempo_Computazione = (Tempo_Base / Compute_Power) * Variazione_Casuale(0.9, 1.1)
Tempo_Trasmissione = (Dimensione_Dati_MB / Bandwidth) * Jitter_Rete(0.8, 1.2)
```

Questa implementazione permette di testare il comportamento del sistema sotto condizioni di rete variabili e con client dalle prestazioni eterogenee.

---

## Politiche di Aggregazione

### Politiche Implementate

**UniformAggregation (FedAvg Classico)**
Implementa l'aggregazione federata standard dove tutti i client contribuiscono equamente:

- Peso uniforme per tutti i partecipanti
- Equivalente al FedAvg originale
- Adatto per scenari con client omogenei

**PowerAwareAggregation**
Favorisce client con maggiore potenza computazionale:

- Pesi proporzionali alla capacità di calcolo
- Migliora la convergenza in scenari eterogenei
- Riduce l'impatto di client lenti

**ReliabilityAwareAggregation**
Considera l'affidabilità storica dei client:

- Maggiore peso ai client più stabili
- Riduce l'impatto di client intermittenti
- Migliora la robustezza del training

**BandwidthAwareAggregation**
Ottimizza per le condizioni di rete:

- Considera la larghezza di banda disponibile
- Filtra client con connessioni inadeguate
- Soglia minima configurabile (default 3.0 Mbps)

**HybridAggregation**
Combina multiple metriche per la selezione ottimale:

- Algoritmo composito che bilancia potenza, affidabilità e bandwidth
- Selezione adattiva basata su score combinato
- Massima flessibilità per scenari complessi

### Aggregazione Parziale

Il sistema supporta l'aggregazione anche quando non tutti i client rispondono:

- Soglia minima configurabile tramite `min_clients_for_aggregation`
- Distribuzione universale del modello aggiornato a tutti i client registrati
- Gestione automatica dei timeout per evitare blocchi indefiniti
- Fallback a strategie alternative quando insufficienti client rispondono

---

## Algoritmi di Federated Learning

### Algoritmi Supportati

**FedAvg (Federated Averaging)**
L'algoritmo fondamentale del federated learning:

- Media pesata dei parametri del modello
- Implementazione ottimizzata per reti neurali profonde
- Baseline per confronti di performance

**FedProx (Federated Proximal)**
Estensione robusta di FedAvg con termine prossimale:

- Aggiunge regolarizzazione per gestire l'eterogeneità dei dati
- Parametro `proximal_term` configurabile (default: 0.01)
- Migliore convergenza in scenari non-IID

**FedYogi**
Ottimizzazione adattiva con controllo del momentum:

- Algoritmo di ottimizzazione server-side adattivo
- Parametri configurabili: `beta1`, `beta2`, `eta`, `tau`
- Convergenza accelerata e stabilità migliorata

**FedAdam**
Variante federata dell'ottimizzatore Adam:

- Adatta l'algoritmo Adam al contesto federato
- Stessi parametri di FedYogi con implementazione diversa
- Efficace per modelli con molti parametri

### Configurazione degli Algoritmi

Gli algoritmi sono configurati nel file `cfg/config.json`:

```json
{
  "fl_algorithm": "fedyogi",
  "proximal_term": 0.01,
  "beta1": 0.9,
  "beta2": 0.99,
  "eta": 0.01,
  "tau": 0.001
}
```

---

## Sistemi di Testing

### Sistema Principale (start_server.py + start_clients.py)

Architettura basata su processi separati per massimo controllo:

**Caratteristiche:**

- Controllo granulare di server e client
- Sistema moderno e flessibile
- Supporto per tutte le politiche di aggregazione
- Simulazione risorse avanzata configurabile

**Utilizzo:**

```bash
# Terminal 1: Server
./venv/Scripts/python.exe start_server.py --policy uniform

# Terminal 2: Client
./venv/Scripts/python.exe start_clients.py --mode resource --policy power
```

#### Sistema Principale - Web App version
È stata sviluppata una **Web Application basata su Gradio** che permette di modificare facilmente il file di configurazione e visualizzare i risultati sia del server che dei client, senza dover utilizzare il terminale. L’interfaccia, infatti, è **user-friendly** e offre diverse modalità di visualizzazione.

Per avviare l’applicazione è sufficiente eseguire il file `gradio_app.py`.  
Le sezioni disponibili sono:

- **Configurazione**: consente di modificare in modo semplice il file `config.json`, facilitando la sperimentazione con diversi modelli, client e parametri di training.
- **Avvio Server/Client**: permette di avviare il server e i client, oltre a selezionare la tipologia di aggregazione dei risultati.
- **Log & Output**: mostra l’output del terminale di server e client, con la possibilità di aggiornare lo stato tramite un apposito tasto.
- **Metriche**: presenta in formato tabellare le metriche di tutti i client per round, con un pulsante dedicato per aggiornarne i valori nel tempo.
- **Dashboard**: fornisce una visualizzazione grafica, oltre a quella tabellare, per monitorare l’andamento delle metriche round dopo round.
- **Server Training Metrics**: mostra le metriche aggregate dal server sia in formato tabellare che grafico, con un apposito tasto per l’aggiornamento dei dati.

### Sistema Automatico Policy Testing

Sistema per testing automatico di multiple politiche di aggregazione:

**Caratteristiche:**

- Automazione completa con server e client integrati
- Testing batch di tutte le politiche disponibili
- Simulazione risorse diversificate per ogni client
- Confronto sistematico tra approcci di aggregazione

**Utilizzo:**

```bash
# Test policy specifica
./venv/Scripts/python.exe run_multiple_clients_with_parameters.py --policy power

# Test tutte le policy
./venv/Scripts/python.exe run_multiple_clients_with_parameters.py --policy all
```

### Sistema Framework Comparison

Sistema di confronto automatico tra framework custom e FLOWER:

**Caratteristiche:**

- Confronto sistematico delle performance
- Output organizzato in struttura modulare
- Report dettagliati con metriche comparative
- Testing di configurazioni multiple automatico

**Utilizzo:**

```bash
# Test rapido
echo "1" | ./venv/Scripts/python.exe testing_frameworkcustom_vs_FLOWER.py

# Test completo
echo "2" | ./venv/Scripts/python.exe testing_frameworkcustom_vs_FLOWER.py
```

### Sistema Client Legacy

Sistema semplificato per client automatici con server esterno:

**Caratteristiche:**

- Client automatici per server già avviato
- Compatibilità con implementazioni precedenti
- Setup semplice per testing rapidi

**Utilizzo:**

```bash
# Prima avvia server manualmente
./venv/Scripts/python.exe start_server.py --policy uniform

# Poi client automatici
./venv/Scripts/python.exe run_multiple_clients.py
```

---

## Configurazione del Sistema

### File di Configurazione Principale

Il file `cfg/config.json` controlla tutti gli aspetti del sistema:

**Connettività:**

```json
{
  "ip_address": "127.0.0.1",
  "port": 5000
}
```

**Parametri di Training:**

```json
{
  "global_epoch": 2,
  "local_epoch": 1,
  "num_clients": 4,
  "learning_rate": 1e-5,
  "batch_size": 16
}
```

**Aggregazione Parziale:**

```json
{
  "partial_aggregation_enabled": true,
  "min_clients_for_aggregation": 2,
  "client_response_timeout": 60
}
```

**Algoritmo FL:**

```json
{
  "fl_algorithm": "fedyogi",
  "proximal_term": 0.01,
  "beta1": 0.9,
  "beta2": 0.99,
  "eta": 0.01,
  "tau": 0.001
}
```

### Struttura delle Directory

**Output del Framework Comparison:**

```
framework_comparison_outputs/
├── results/     # File CSV con metriche dettagliate
├── logs/        # Log di esecuzione completi
└── summaries/   # Report riassuntivi comparativi
```

**Log del Sistema:**

```
logs/
└── MMDD/
    ├── FL-Server-LOG/  # Log del server
    └── FL-Client-LOG/  # Log dei client
```

**Risultati CSV:**

```
csv/
└── MMDD/              # Metriche di training per data
```

---

## Monitoraggio e Debugging

### Sistema di Logging

Il sistema implementa logging strutturato multi-livello:

- **Server Logs**: Decisioni di aggregazione, selezione client, metriche globali
- **Client Logs**: Training locale, risorse utilizzate, comunicazione
- **Framework Logs**: Confronti di performance, errori di sistema

### Metriche Monitorate

**Metriche di Training:**

- Loss di training e validazione per round
- Accuracy, Precision, Recall, F1-Score
- Tempo di convergenza
- Numero di client partecipanti per round

**Metriche di Sistema:**

- Utilizzo delle risorse simulate
- Tempi di comunicazione e computazione
- Tasso di successo dei client
- Distribuzione dei pesi di aggregazione
