# Streaming benchmark (synthetic nights from the bench partition)

FSM params: `{"tau": 0.8060498833656311, "tick_ms": 500, "hold_ticks": 12, "confirm_ticks": 20, "verify_ticks": 30, "min_bursts": 3, "period_min_ticks": 3, "period_max_ticks": 14}`

## predictor: float

| SNR dB | nights | detection | confirm latency s | false confirms / h | stop latency s |
|---|---|---|---|---|---|
| 0 | 5 | 0.044 | 65.4 | 0.60 | 5.3 |
| 5 | 5 | 0.057 | 14.8 | 0.00 | 0.0 |
| 10 | 5 | 0.158 | 43.1 | 0.00 | 0.0 |
| 20 | 5 | 0.271 | 50.6 | 0.20 | 0.0 |

## predictor: int8

| SNR dB | nights | detection | confirm latency s | false confirms / h | stop latency s |
|---|---|---|---|---|---|
| 0 | 5 | 0.044 | 52.6 | 0.60 | 2.3 |
| 5 | 5 | 0.057 | 14.8 | 0.00 | 0.0 |
| 10 | 5 | 0.183 | 41.8 | 0.00 | 0.0 |
| 20 | 5 | 0.293 | 50.1 | 0.20 | 0.2 |

tick decision agreement float~int8: 0.9996; episode starts {'float': 26, 'int8': 28}
