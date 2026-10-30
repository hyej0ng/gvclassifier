# main1_10011716 결과 및 leakage 검토

검토일: 2026-10-08 (한국 시간). 대상: `results/main1_10011716/main`.

## 판단

높은 genome 점수와 완만하게 움직이는 validation loss는 모순이 아니다. 실제 기록은 **label smoothing을 적용한 genome/class 가중 chunk loss**와 **길이 가중 집계 후 genome macro-F1**이라는 서로 다른 목표를 비교하고 있다. 현재 확인한 결과는 genome 단위 내부 holdout 성능으로 해석하는 것이 적절하다.

저장된 예측의 지표를 독립적으로 재계산했고 기존 confusion matrix 및 macro-F1과 정확히 일치했다. split/평가 코드에서 정답을 입력 특징으로 전달하거나 inference split으로 학습·checkpoint 선택·temperature fitting을 하는 경로는 발견하지 못했다. **전체 4,427,636개 chunk의 split 간 exact/역상보 중복은 0개**, genome/group/record ID 교집합도 모두 0개였고 모델·manifest·세 CSV의 SHA-256은 당시 기록과 모두 일치했다. 수치는 `audit_results.json`에 기록한다.

다만 이것은 모든 종류의 leakage가 없다는 증명은 아니다. 특히 foundation model 사전학습 데이터와의 중복, 근연 서열에 대한 유사도 검색의 누락, 같은 출처 내 분할에서 생기는 source 편향은 이번 검사의 범위 밖에 남는다.

## 1. loss가 크게 줄지 않아 보이는 이유

학습 설정은 현재 `configs/pipeline.yaml`이 아니라 당시 저장된 `runs/main/main1_10011716/run_metadata.json`을 기준으로 확인했다. 당시 learning rate는 **1e-5**이며 현재 설정의 **2e-6**과 다르다. label smoothing은 **0.05**, genome balancing은 활성화, class weight는 Cellular **1.45494**, Viral **0.54506**이었다.

| 평가 시점 | train loss | validation loss | validation chunk macro-F1 | validation genome macro-F1 |
|---|---:|---:|---:|---:|
| 0.2 epoch | 0.2180 | 0.16938 | 0.95992 | 0.96806 |
| 2.4 epoch, best | 0.1407 | 0.15011 | 0.91661 | 0.98054 |
| 4.4 epoch, 종료 | 0.1291 | 0.16281 | 0.92041 | 0.97380 |

첫 로그는 학습 전 baseline이 아니다. 약 72만 training chunk를 처리한 뒤의 첫 평가이므로 이미 높은 점수가 나왔다고 해서 학습 전부터 같은 성능이었다고 판단할 수 없다.

### smoothing 때문에 loss의 최솟값이 0이 아니다

PyTorch의 2-class label smoothing 0.05는 정답 target을 `[0.975, 0.025]`로 만든다. 이 target에 대한 cross-entropy의 이론적 최솟값은 다음과 같다.

`−0.975 log(0.975) − 0.025 log(0.025) = 0.11690685`

전체 split에서 sample weight의 평균을 1로 정규화하므로 동일한 하한이 weighted mean loss에도 적용된다. 따라서 validation loss 0.15를 '0에 비해 높다'고 해석하면 안 된다. 첫 0.16938에서 최저 0.15011로 약 11.4% 감소했고, 하한을 초과하는 부분은 약 36.7% 감소했다. 반면 best 이후에는 loss가 다시 상승하고 train loss는 계속 줄어, 추가 학습의 이득이 없거나 과적합이 진행되는 패턴이다.

### loss와 F1이 측정하는 대상이 다르다

- loss: 각 chunk의 confidence까지 평가하는 smoothed cross-entropy. `valid_length/genome_total_bp` 및 train에서 계산한 class weight를 곱한다. 큰 genome에서 수많은 chunk가 나와도 해당 genome의 총 기여도를 제한한다.
- genome F1: 각 genome의 chunk log-probability를 유효 염기 수로 가중 평균하고 마지막에 class를 결정한다. 몇 개의 chunk를 틀려도 genome 전체는 맞힐 수 있다.
- loss는 확률값이 변하면 계속 움직인다. F1은 최종 argmax가 바뀌지 않는 한 변하지 않는다.

best checkpoint는 **validation genome macro-F1**으로 선택한다. `checkpoints/best/best_metric.json`에서 epoch 2.40005, F1 0.980537을 확인했다. 이번 run에서는 validation loss 최저점도 같은 checkpoint에 있지만, 코드의 선택 기준 자체는 loss가 아니다.

특히 validation chunk macro-F1은 첫 0.95992에서 best의 0.91661로 **하락**했다. genome 성능의 개선을 chunk 성능 개선으로 해석하면 안 된다. genome balancing을 적용한 목적함수와 많은 chunk를 가진 genome의 오류 양상이 이 차이에 영향을 줄 수 있다. epoch별 validation 예측이 저장되어 있지 않으므로 악화된 chunk가 어느 source에 집중되는지는 확정하지 못했다.

## 2. 좋은 결과의 실제 범위

| 단위 | 평가 수 | macro-F1 | accuracy |
|---|---:|---:|---:|
| Chunk | 406,682 | 0.94756 | 0.95371 |
| Contig | 69,259 | 0.94429 | 0.94575 |
| Genome | 2,335 | 0.98300 | 0.99186 |

Genome confusion matrix는 실제 class를 행, 예측 class를 열로 두면 `[[315, 0], [19, 2001]]`이다. Cellular 315개는 모두 맞고 Viral 2,020개 중 19개를 놓친다. 따라서 genome Viral recall은 **99.059%**, precision은 **100%**다. 저장된 split-group bootstrap macro-F1 95% CI는 **0.97589–0.99105**다. 이 CI는 정해진 내부 데이터와 split-group 재표집에 대한 불확실성이며 외부 출처 성능까지 보장하지 않는다.

특히 Cellular 쪽에서 집계 효과가 크다. EUK의 chunk accuracy는 **87.802%**, contig accuracy는 **82.402%**지만 genome은 **12/12 정답**이다. Cellular 전체 chunk에서 Viral로 잘못 분류된 조각이 **17,739개**나 있다. 따라서 genome 결과만 보고 실제 5 kb 조각이나 짧은 contig에서 false positive가 거의 없다고 결론 내리면 안 된다. chunk Viral precision은 **87.509%**다.

| Viral source | genome 정답/전체 | recall |
|---|---:|---:|
| NCLDV | 1,547/1,549 | 99.871% |
| Mirus | 157/157 | 100% |
| PHAGE | 250/259 | 96.525% |
| EVE NCLDV | 38/45 | 84.444% |
| EVE Mirus | 9/10 | 90% |

NCLDV가 전체 genome의 **66.3%**이고 Viral genome의 **76.7%**다. main macro-F1은 Cellular/Viral 두 class에 동일 비중을 주지만 Viral 내부의 source별 비중까지 균등하게 만들지는 않는다. EVE의 상대적으로 낮은 성능이 전체 점수에 작게 반영된다. 10개 source의 genome accuracy를 동일 비중으로 평균하면 **97.084%**다. 이 값은 main macro-F1과 다른 보조 지표다.

`genome_errors.csv`에 19개 오류를 저장했다. 오류 구성은 PHAGE 9, EVE NCLDV 7, NCLDV 2, EVE Mirus 1이다.

EVE에서 `genome_id`는 viral genome이 아니라 **동일 host assembly의 EVE locus 묶음**이다. 따라서 EVE genome 지표는 host별 집계 지표이며 개별 EVE locus 탐지 성능으로 그대로 읽으면 안 된다. EVE NCLDV contig recall은 **50/59 = 84.746%**, EVE Mirus는 **14/16 = 87.5%**다.

## 3. calibration이나 평가 버그가 점수를 만든 것인가

저장된 chunk probability를 재집계했을 때 chunk/contig/genome의 confusion matrix와 macro-F1 차이는 모두 **0**이다. 평가 코드에서 확률 집계에 정답을 사용하지 않는다. `labels`는 집계 후 metric 계산에만 쓰인다. source/genome ID/header도 tokenizer 입력에 들어가지 않고 DNA sequence만 입력한다.

Temperature는 validation genome NLL만으로 선택한 **0.75**다. 같은 양수 scalar temperature는 각 chunk 및 genome의 argmax를 보존하며, 저장된 확률로 temperature 적용을 되돌렸을 때 실제 예측 변화도 **0개**였다. 따라서 높은 F1의 원인은 temperature가 아니다.

확률 품질에는 영향을 준다. holdout genome NLL은 calibration 전 약 **0.04921**, 후 **0.03609**로 개선되지만 chunk NLL은 약 **0.15896 → 0.16861**로 악화된다. genome을 대상으로 calibration했으므로 chunk의 confidence 품질이 좋아진다고 보장할 수 없다.

저장된 확률에서 temperature를 되돌리고 당시 가중치와 smoothing을 적용해 근사 재계산한 holdout chunk loss는 **0.14927**이다. best validation loss **0.15011**과 매우 가깝다. 즉, holdout에서도 동일 loss 관점으로 보면 비슷한 수준이며, 'validation은 나쁜데 holdout만 비정상적으로 좋다'는 모습은 아니다. 저장된 float32 확률·clipping에서 역산한 값이므로 모델 forward를 다시 실행한 정확한 loss와는 구분한다.

## 4. leakage 검토

### 직접 검증하는 사항

`audit.py`는 원본 결과를 변경하지 않고 다음 항목을 검사하며 수치를 `audit_results.json`에 저장한다.

1. 원본 manifest, 당시 best.pt 및 세 split CSV의 SHA-256을 다시 계산하고 저장된 fingerprint와 비교.
2. manifest에서 genome ID, base group, similarity component, 전체 record의 canonical sequence hash가 split을 넘나드는지 검사.
3. 기록된 유사도 edge **10,243개** 중 양쪽 group에 사용 가능한 record가 남아 있는 **10,239개**가 같은 split인지 검사. **cross-split edge 0개**.
4. main의 모든 **4,427,636개 chunk**를 다시 읽어 canonical sequence hash를 계산. 순방향·역상보를 동일하게 취급하고 train/validation/inference 교차 중복 확인.
5. 각 chunk의 record ID, genome ID, label, split이 manifest와 일치하는지 확인하고 inference 예측 row 순서와 정답도 원본 inference CSV와 대조.

기존 preprocessing QC 또한 split-group/genome/contig/exact·역상보 chunk 중복을 검사하며 passed=true다. 이번 검토는 기존 passed 플래그를 그대로 신뢰하는 데 그치지 않고 실제 CSV 및 파일 hash를 다시 검사한다.

최종 재검사 결과는 다음과 같다.

| 항목 | 결과 |
|---|---|
| Train–validation canonical chunk 중복 | 0 |
| Train–inference canonical chunk 중복 | 0 |
| Validation–inference canonical chunk 중복 | 0 |
| 모든 split 쌍의 genome/group/record ID 겹침 | 0 |
| 기록된 active near-similarity edge의 split 교차 | 0 / 10,239 |
| Chunk와 manifest의 split/genome/label 불일치 | 0 / 4,427,636 |
| Inference 예측 row 순서 및 label 대조 | 일치 |
| Manifest, best.pt, train/validation/inference CSV hash | 모두 일치 |

유효한 NCLDV genus **2,734개**는 split 간 공유가 **0개**다. 하지만 taxonomy가 없거나 사용할 수 없는 genus를 가진 NCLDV도 있다. inference 1,549개 중 **1,237개(79.9%)**만 genus 단위 group에 속하고 **312개(20.1%)**는 genome/similarity grouping이다. 전체 NCLDV가 모두 genus-disjoint라고 표현하면 부정확하다. genus-grouped holdout은 1,237/1,237 정답, 나머지는 310/312 정답이다. `g_`, 빈 이름 등은 실제 genus 공유로 세지 않았다.

EVE host assembly 및 PHAGE vOTU도 source 내 split 간 공유가 없다. EVE NCLDV/Mirus는 코드상 같은 `EVE_HOST` base group을 사용하므로 양쪽에 등장한 동일 host도 함께 배정된다.

### 남는 불확실성

**A. Foundation model 사전학습과의 중복**

현재 공개 [GenomeOcean v1.2 model card](https://huggingface.co/DOEJGI/GenomeOcean-100M-v1.2/blob/main/README.md?code=true)는 original metagenomic corpus에 GTDB r226, INPHARED phage, RNA virus 데이터가 추가되었다고 설명한다. 이번 파이프라인에는 해당 pretraining corpus와 holdout을 대조·제외하는 절차가 없다. 따라서 fine-tuning split이 깨끗해도 foundation model이 이미 본 서열을 평가에 포함했을 가능성은 남는다. 이것은 이번 fine-tuning에서 정답을 유출했다는 증거와는 구별해야 한다.

또한 현재 model card에는 초기에 base-model weights를 잘못 올렸다가 continued-training weights로 정정했다는 설명이 있다. 이번 run은 revision `26dd3863d7e66a1e1d87ae4e45968bfdc1fa098f`를 고정하고 실제 commit도 기록한다. **현재 main의 학습 corpus 설명을 해당 pinned revision의 실제 weights에 그대로 적용할 수는 없다.** 이번에는 pinned revision의 pretraining accession/sequence 목록과 weight 계보를 확보하지 못해 중복 여부를 확정하지 못했다. 확인되지 않은 overlap 수치를 추정해 쓰지 않았다.

**B. Near-duplicate 검사 범위**

preprocessing은 5 kb fragment에 MMseqs Linclust identity 95%, 양방향 coverage 85%, Mirus에는 추가 genome ANI/AF를 적용한다. 기록된 연결은 잘 차단되어 있다. 하지만 Linclust 후보 탐색은 모든 가능한 local alignment를 빠짐없이 조사하는 검사가 아니다. 경계가 어긋난 homologous fragment, coverage가 낮은 부분 공유, 기준 이하의 근연 서열까지 제거한 것은 아니다. 이번 검토에서는 전체 near-similarity 검색을 다시 실행하지 않았다.

**C. Source 및 taxonomy 일반화**

각 source 내부에서 80/10/10을 나누므로 출처별 assembly/GC/서열 품질 등의 특성을 이용해도 내부 holdout에서 좋은 점수가 날 수 있다. 모델 입력에 source 문자열이 들어가는 직접 누수는 없지만, 생물학적 특징과 출처 특성이 혼재할 수 있다. 현재 자료만으로 그 기여도를 분리할 수 없다. Cellular/Mirus 등 전체 데이터에 대한 genus/family/study-disjoint 평가도 아니다.

**D. Holdout 반복 사용**

코드상 best 선택과 temperature fitting은 validation만 사용한다. 다만 사람이 main1 결과를 본 뒤 main2 설정을 바꾸거나 여러 run 중 holdout 점수로 최종 모델을 고르면 같은 inference split은 개발 과정에 영향을 준다. 실제 의사결정 과정은 기록만으로 확정할 수 없다. 그런 경우 최종 주장에는 별도의 untouched test set이 필요하다.

## 5. 권장 후속 평가

1. **우선 목표 단위를 정한다.** genome 분류가 목표면 현재 best를 유지할 근거가 있다. 5 kb/contig 탐지가 목표면 chunk macro-F1/PR-AUC/Cellular false-positive rate를 checkpoint 선택 기준으로 삼는 별도 run이 필요하다. 지금 inference 결과로 checkpoint를 선택하지 말고 validation에서 결정한다.
2. **학습률 변경보다 외부 평가를 우선한다.** 새 source/study 또는 더 엄격한 taxonomy holdout을 고정하고, Cellular의 짧은 조각과 EVE를 별도로 평가한다. 높은 genome 점수가 실제 사용 목적에 옮겨가는지 검증한다.
3. **Pretraining overlap을 별도로 확인한다.** 먼저 pinned revision의 실제 weight와 corpus를 확정한 뒤 holdout accession·sequence를 비교한다. 새로 생성한 서열이라도 원래 같은 assembly에서 파생됐다면 날짜만으로 독립성을 보장하지 않는다.
4. **Source shortcut을 측정한다.** train에서만 fit한 GC/N 비율/길이/k-mer baseline을 고정된 validation/외부 평가에 비교한다. source를 하나씩 통째로 제외하는 stress test도 도움이 된다.
5. **오류 19개와 EUK false positive를 생물학적으로 검토한다.** EVE boundary의 host flank, PHAGE snapshot의 품질·비세균 숙주 viral group, NCLDV bin contamination 등을 후보로 확인하되 현재 오류를 label 오류로 단정하지 않는다.

이번 검토에서는 모델 재학습, 원본 데이터 재분할, 기존 결과 덮어쓰기를 수행하지 않았다. 검토 산출물만 이 디렉터리에 추가했다.

## 산출물

- `audit_results.json`: 지표 재계산, manifest/edge/파일 hash 및 전체 chunk 중복 검사.
- `audit.py`, `audit.log`: 재현 가능한 검토 코드와 실행 기록.
- `supplement.py`, `supplement_results.json`: 유효 genus 구분 및 NCLDV subgroup 점수.
- `validation_detailed.csv`: trainer_state에서 추출한 epoch별 chunk/genome validation 지표.
- `holdout_loss_decomposition.json`: 동일 목적함수로 역산한 holdout loss.
- `loss_vs_f1.png`: loss와 chunk/genome F1의 추이.
- `genome_errors.csv`: genome 오류 19개.
