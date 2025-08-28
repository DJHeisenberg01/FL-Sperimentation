# Sistema di Federated Learning Resource-Aware

## Documentazione Tecnica Completa

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

### Nuova Gestione dei Profili Client

Il sistema ora supporta la configurazione avanzata dei client tramite profili predefiniti che sostituiscono la generazione completamente casuale. Questa funzionalità permette di controllare precisamente le caratteristiche dei client per esperimenti riproducibili.

### Profili di Qualità Disponibili

**Profilo "ottimali"**
Client con risorse elevate ideali per testing di performance:

- Compute Power: 1.5-2.0 (Workstation/Server con GPU dedicate)
- Bandwidth: 8.0-10.0 Mbps (Fibra ottica/5G enterprise)
- Reliability: 0.95-1.0 (Quasi sempre disponibili)

**Profilo "bilanciati" (Default)**
Mix equilibrato di client con risorse moderate:

- Compute Power: 0.8-1.5 (Desktop standard a workstation)
- Bandwidth: 4.0-8.0 Mbps (4G+ a fibra domestica)
- Reliability: 0.85-0.95 (Abbastanza affidabili)

**Profilo "scarsi"**
Client con risorse limitate per test di robustezza:

- Compute Power: 0.3-0.8 (IoT/dispositivi mobili)
- Bandwidth: 1.0-4.0 Mbps (3G/WiFi lento)
- Reliability: 0.7-0.85 (Frequenti disconnessioni)

**Profilo "misti"**
Distribuzione eterogenea di tutti i tipi di client:

- 20% client ottimali
- 60% client bilanciati
- 20% client scarsi
- Distribuzione personalizzabile in configurazione

### Configurazione nel File config.json

Il file `cfg/config.json` include ora la sezione `client_configuration`:

```json
{
  "client_configuration": {
    "num_clients": 4,
    "client_quality": "bilanciati",
    "mix_distribution": {
      "ottimali": 0.2,
      "bilanciati": 0.6,
      "scarsi": 0.2
    }
  }
}
```

**Parametri disponibili:**

- `num_clients`: Numero di client da generare (sovrascrive il campo legacy)
- `client_quality`: Profilo di qualità (`"ottimali"`, `"bilanciati"`, `"scarsi"`, `"misti"`)
- `mix_distribution`: Distribuzione personalizzata per profilo "misti"

### Utilizzo della Nuova Gestione

!! IMPORTANTE !!
Bisogna sempre utilizzare prima questa funzione per modificare il file di config, per poi andare a chiamare start_server ecc.

**Modifica Profilo Tramite Script di Utilità:**

```bash
# Imposta client ottimali
python set_client_profile.py --profile ottimali

# Imposta client scarsi con 6 client
python set_client_profile.py --profile scarsi --clients 6

# Imposta distribuzione mista con 8 client
python set_client_profile.py --profile misti --clients 8

# Visualizza configurazione attuale
python set_client_profile.py --show
```

**Modifica Manuale nel config.json:**

```json
{
  "client_configuration": {
    "num_clients": 6,
    "client_quality": "ottimali"
  }
}
```

**Testing dei Profili:**

```bash
# Testa tutti i profili e funzionalità
python test_client_profiles.py
```

### Classe ClientResources

La simulazione delle risorse è implementata attraverso la classe ClientResources che modella tre aspetti fondamentali:

**Compute Power (Potenza Computazionale)**
Rappresenta la capacità di calcolo del client relativa a un dispositivo baseline:

- 0.3-0.8: Dispositivi IoT e mobili con hardware limitato
- 0.8-1.5: Desktop standard a workstation moderne
- 1.5-2.0: Server e workstation con GPU dedicate

**Bandwidth (Larghezza di Banda)**
Velocità di connessione di rete del client in Megabit per secondo:

- 1.0-4.0 Mbps: Connessioni lente (3G, WiFi debole)
- 4.0-8.0 Mbps: Connessioni moderate (4G, WiFi domestico)
- 8.0-10.0 Mbps: Connessioni ottimali (5G, Fibra ad alta velocità)

**Reliability (Affidabilità)**
Probabilità che il client completi con successo un round di training:

- 0.7-0.85: Client instabili con frequenti disconnessioni
- 0.85-0.95: Client moderatamente affidabili
- 0.95-1.0: Client altamente stabili e affidabili

### Simulazione Realistica

Il sistema introduce variabilità stocastica per simulare condizioni reali:

```
Tempo_Computazione = (Tempo_Base / Compute_Power) * Variazione_Casuale(0.9, 1.1)
Tempo_Trasmissione = (Dimensione_Dati_MB / Bandwidth) * Jitter_Rete(0.8, 1.2)
```

Questa implementazione permette di testare il comportamento del sistema sotto condizioni di rete variabili e con client dalle prestazioni eterogenee, mantenendo però la riproducibilità degli esperimenti tramite i profili predefiniti.

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

### Analisi delle Strategie di Aggregazione

Durante lo sviluppo del sistema, sono stati valutati due approcci principali per gestire l'aggregazione dei modelli:

#### Approccio 1: Aggregazione Dinamica Multipla (Implementazione Iniziale)

**Concetto**: Eseguire l'aggregazione ogni volta che arriva una nuova risposta da un client, aggiornando continuamente il modello globale durante lo stesso round.

**Vantaggi teorici**:

- Incorporazione immediata degli aggiornamenti dei client
- Potenziale convergenza più rapida in alcuni scenari
- Adattamento dinamico ai pattern di risposta dei client

**Problemi identificati**:

- **Inefficienza computazionale**: Re-aggregazione completa per ogni nuova risposta
- **Inconsistenza del modello**: Il modello globale cambia più volte nello stesso round
- **Spreco di risorse**: I contributi dei client che rispondono prima vengono "sovrascritti"
- **Complessità di gestione**: Difficile determinare lo stato "finale" di ogni round
- **Race conditions potenziali**: Aggregazioni concorrenti possono portare a stati inconsistenti

#### Approccio 2: Aggregazione Timer-Based Singola (Implementazione Attuale)

**Concetto**: Attendere un periodo configurabile per raccogliere il maggior numero possibile di risposte dei client, quindi eseguire una singola aggregazione per round.

**Vantaggi dell'implementazione**:

- **Efficienza computazionale**: Una sola operazione di aggregazione per round
- **Consistenza del modello**: Un unico modello globale definitivo per round
- **Massimizzazione della partecipazione**: Attesa di tutti i client disponibili prima del timeout
- **Allineamento con le best practices**: Conforme ai protocolli standard del federated learning
- **Comportamento predicibile**: Confini chiari dei round e stati del modello

**Parametri di configurazione**:

- `partial_aggregation_wait_time`: Tempo di attesa prima dell'aggregazione parziale (default: 30 secondi)
- `min_clients_for_aggregation`: Numero minimo di client richiesti per procedere
- `client_response_timeout`: Timeout massimo per le risposte dei client

**Meccanismo di timeout implementato**:

1. Quando viene raggiunto il numero minimo di client, viene avviato un timer
2. Se tutti i client rispondono prima del timeout, il timer viene cancellato e si procede immediatamente
3. Se il timeout scade, si procede con l'aggregazione parziale usando i client disponibili
4. Gestione robusta dei casi edge e prevenzione delle aggregazioni multiple

**Scelta progettuale**: È stato selezionato l'approccio timer-based per la sua efficienza, consistenza e allineamento con le pratiche consolidate del federated learning.

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
# Terminal 1: Server con configurazione da config.json
./venv/Scripts/python.exe start_server.py --policy uniform

# Terminal 2: Client (legge automaticamente num_clients e profilo da config.json)
./venv/Scripts/python.exe start_clients.py --mode resource --policy power

# Per usare profili client specifici:
# 1. Prima configura il profilo
python set_client_profile.py --profile ottimali --clients 6

# 2. Poi avvia server e client (useranno automaticamente la configurazione)
./venv/Scripts/python.exe start_server.py --policy power
./venv/Scripts/python.exe start_clients.py --mode resource --policy power
```

### Sistema Automatico Policy Testing (⚠️ Problemi Noti di Threading)

> **⚠️ ATTENZIONE CRITICA**: Il sistema automatico `run_multiple_clients_with_parameters.py` presenta problemi significativi di race conditions e sincronizzazione thread che lo rendono **instabile per esecuzioni di produzione**.

Sistema per testing automatico di multiple politiche di aggregazione:

**Caratteristiche:**

- Automazione completa con server e client integrati
- Testing batch di tutte le politiche disponibili
- Simulazione risorse diversificate per ogni client
- Confronto sistematico tra approcci di aggregazione

**⚠️ Problemi identificati**:

- **Race Conditions**: Thread multipli accedono concorrentemente al server federato
- **Sincronizzazione inconsistente**: Conflitti durante l'aggregazione dei modelli
- **Esecuzione non deterministica**: I round possono terminare prematuramente o duplicarsi
- **Perdita di coordinazione client**: Disconnessioni e riconnessioni inconsistenti
- **Risultati non riproducibili**: Variabilità nei risultati tra esecuzioni multiple

**Sintomi osservabili**:

- Round che si completano con meno client del previsto
- Aggregazioni multiple nello stesso round
- Client che perdono sincronizzazione con il server
- Timeout inaspettati e fallimenti di comunicazione
- Log inconsistenti tra esecuzioni

**Utilizzo (⚠️ Solo per sviluppo/debugging)**:

```bash
# ⚠️ Test policy specifica - INSTABILE
./venv/Scripts/python.exe run_multiple_clients_with_parameters.py --policy power

# ⚠️ Test tutte le policy - INSTABILE
./venv/Scripts/python.exe run_multiple_clients_with_parameters.py --policy all

# Per testare con profili specifici:
# 1. Configura il profilo desiderato
python set_client_profile.py --profile scarsi --clients 6

# 2. Esegui il test (userà automaticamente la configurazione)
./venv/Scripts/python.exe run_multiple_clients_with_parameters.py --policy power
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
# Test rapido con configurazione da config.json
# 2. Esegui il test di confronto (userà automaticamente la configurazione)
echo "1" | ./venv/Scripts/python.exe testing_frameworkcustom_vs_FLOWER.py
```

### Raccomandazioni Architetturali

#### ✅ Approccio Raccomandato: Process-Based Execution

**Per esecuzioni di produzione e testing affidabile**:

```bash
# Terminal 1 - Server
python start_server.py --policy <policy_name> --config cfg/config.json

# Terminal 2 - Clients
python start_clients.py --mode resource --policy <policy_name> --num-clients <N>
```

**Vantaggi**:

- **Isolamento completo**: Processi separati eliminano race conditions
- **Simulazione realistica**: Rispecchia meglio un ambiente federato distribuito
- **Debugging facilitato**: Log separati e gestione errori indipendente
- **Scalabilità**: Facile distribuzione su macchine multiple
- **Stabilità**: Risultati consistenti e riproducibili

#### ⚠️ Approccio Deprecato: Thread-Based Execution

**Solo per sviluppo rapido o prototipazione**:

```bash
# ⚠️ INSTABILE - Solo per sviluppo
python run_multiple_clients_with_parameters.py --policy <policy_name>
```

**Limitazioni**:

- Race conditions inevitabili in scenari complessi
- Difficile debugging di problemi di concorrenza
- Risultati non deterministici
- Non rappresentativo di deployment reale

#### Linee Guida per lo Sviluppo

1. **Sempre testare con process-based execution** prima di considerare completate le feature
2. **Utilizzare thread-based execution solo per prototipazione rapida** di nuove funzionalità
3. **Documentare esplicitamente** eventuali limitazioni nei commit che toccano il threading
4. **Prioritizzare la stabilità** sull'automazione quando c'è conflitto tra i due obiettivi

---

# Test completo con configurazione avanzata

echo "2" | ./venv/Scripts/python.exe testing_frameworkcustom_vs_FLOWER.py

# Per testare con profili client specifici:

# 1. Configura prima il profilo desiderato

python set_client_profile.py --profile misti --clients 8

# 2. Esegui il test di confronto (userà automaticamente la configurazione)

echo "2" | ./venv/Scripts/python.exe testing_frameworkcustom_vs_FLOWER.py

````

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
````

### Gestione Profili Client

**Script di Utilità `set_client_profile.py`**

Utility per cambiare facilmente il profilo client in config.json:

```bash
# Imposta profilo ottimali (mantiene numero client esistente)
python set_client_profile.py --profile ottimali

# Imposta profilo scarsi con 6 client
python set_client_profile.py --profile scarsi --clients 6

# Imposta distribuzione mista con 8 client
python set_client_profile.py --profile misti --clients 8

# Mostra configurazione attuale senza modificarla
python set_client_profile.py --show

# Esempio con percorso config personalizzato
python set_client_profile.py --profile bilanciati --config cfg/custom_config.json
```

**Testing e Validazione**

```bash
# Test completo di tutti i profili e funzionalità
python test_client_profiles.py

# Verifica che la configurazione sia valida
python -c "from client_resources import ClientResources; print(ClientResources.validate_config({'client_configuration': {'num_clients': 4, 'client_quality': 'bilanciati'}}))"
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
  "learning_rate": 1e-5,
  "batch_size": 16
}
```

**Configurazione Client (Nuova):**

```json
{
  "client_configuration": {
    "num_clients": 4,
    "client_quality": "bilanciati",
    "mix_distribution": {
      "ottimali": 0.2,
      "bilanciati": 0.6,
      "scarsi": 0.2
    }
  }
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

### Interfaccia Web

Interfaccia di monitoraggio disponibile su `http://127.0.0.1:5000/`:

- Visualizzazione real-time delle metriche
- Stato dei client connessi
- Progresso dei round di training
- Grafici di convergenza

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

---

## Risoluzione Problemi Comuni

### Problemi di Connessione

**"Server not running"**

- Verificare che start_server.py sia avviato per primo
- Controllare che la porta 5000 non sia in uso
- Verificare la configurazione IP in config.json

**Client si disconnettono frequentemente**

- Aumentare `client_response_timeout` in config.json
- Verificare i valori di `reliability` nelle risorse simulate
- Controllare la stabilità della rete

### Problemi di Training

**Training non procede**

- Verificare `min_clients_for_aggregation` vs numero client connessi
- Controllare che i dataset siano correttamente suddivisi
- Verificare la configurazione dell'algoritmo FL

**Convergenza lenta**

- Aumentare `learning_rate` con cautela
- Considerare l'uso di FedProx per dati non-IID
- Ottimizzare la policy di aggregazione per il scenario

### Problemi di Performance

**Training troppo lento**

- Ridurre `local_epoch` se appropriato
- Ottimizzare `batch_size` per l'hardware disponibile
- Utilizzare policy PowerAware per favorire client veloci

**Uso eccessivo di memoria**

- Ridurre `batch_size` nei client
- Implementare gradient checkpointing se necessario
- Monitorare l'uso di memoria durante il training

---

## Estensioni e Personalizzazioni

### Aggiunta di Nuove Politiche

Per implementare nuove politiche di aggregazione:

1. Creare classe che eredita da base aggregation policy
2. Implementare metodi `select_clients` e `aggregate_weights`
3. Aggiungere la policy al dizionario in aggregation_policies.py
4. Testare con run_multiple_clients_with_parameters.py

### Implementazione di Nuovi Algoritmi FL

Per aggiungere algoritmi FL:

1. Implementare la logica client-side in federated_client.py
2. Aggiungere parametri specifici in config.json
3. Implementare aggregazione server-side se necessaria
4. Aggiornare la documentazione e i test

### Integrazione di Nuovi Dataset

Per utilizzare dataset personalizzati:

1. Implementare data loader compatibile con la struttura esistente
2. Aggiornare dataset_splitter_clients.py se necessario
3. Configurare i path in config.json
4. Testare la suddivisione federata dei dati

Il sistema è progettato per essere modulare e estensibile, permettendo facili personalizzazioni per scenari specifici di ricerca in federated learning.
