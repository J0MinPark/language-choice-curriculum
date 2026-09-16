"""Render retrospective figures solely from the frozen trajectory JSON (CPU)."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .choice_trajectory import OUT, PHASES, WRAPPERS

COLORS={'ko':'#0072B2','en':'#E69F00','zh':'#009E73','fr':'#CC79A7'}
LABELS=['T0\nKO trained','T1\n+ EN','T2\n+ ZH','T3\n+ FR']


def save(fig, stem):
    for suffix in ('png','pdf','svg'):
        fig.savefig(OUT/f'{stem}.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    s=json.loads((OUT/'trajectory_summary.json').read_text())
    ep={p:next(t for t in s['snapshots'] if t['phase']==p and t['endpoint']) for p in PHASES}
    eight=s['eight_wrapper_endpoints'];ids=sorted(eight['T0']['Q_by_concept'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
    a,b,c,d=axs.flat
    fig.suptitle('Requested Korean production persists as language choice shifts',fontsize=16)
    for lang,color in COLORS.items():
        mean=[];low=[];high=[]
        for p in PHASES:
            vals=[ep[p]['aggregate']['wrappers'][w]['A'].get(lang,np.nan) for w in WRAPPERS]
            mean.append(np.mean(vals));low.append(np.min(vals));high.append(np.max(vals))
        a.plot(range(4),mean,'o-',color=color,label=lang.upper(),lw=2,ms=5)
        a.fill_between(range(4),low,high,color=color,alpha=.12)
        b.plot(range(4),[eight[p]['Q_mean'][lang] for p in PHASES], 'o-',color=color,label=lang.upper(),lw=2)
    a.set(title='A  Requested-language registered production',ylabel='Correct fraction (2 templates)')
    b.set(title='B  Language choice among registered expressions',ylabel='Mean conditional Q (8 ANY templates)')
    a.legend(ncol=4,loc='lower left',fontsize=9)
    b.legend(ncol=4,loc='center left',fontsize=9)
    retained={r['concept_id'] for r in s['matched_concepts'] if r['T3_KO_success_wrappers']==2}
    for i in ids:
        c.plot(range(4),[eight[p]['Q_by_concept'][i]['ko'] for p in PHASES],
               color=COLORS['ko'] if i in retained else '#999999',alpha=.22,lw=.8)
    c.plot(range(4),[eight[p]['Q_mean']['ko'] for p in PHASES], 'o-',color='#222222',lw=2.5,label='All-concept mean')
    c.set(title='C  All 60 concept trajectories',ylabel='Korean Q (8 ANY templates)')
    c.legend(loc='upper right',fontsize=9)
    c.text(.03,.06,'Blue: KO requested success on both templates at T3\nGray: success on fewer than two templates',transform=c.transAxes,fontsize=8.5)
    for count in range(3):
        group=[r for r in s['matched_concepts'] if r['T3_KO_success_wrappers']==count]
        jitter=np.linspace(-.18,.18,len(group)) if len(group)>1 else np.array([0.])
        d.scatter(count+jitter,[r['T3_Q8_FR'] for r in group],s=25,color=COLORS['fr'],alpha=.75,edgecolors='white',linewidths=.4)
        d.text(count,1.045,f'n = {len(group)}',ha='center',fontsize=9)
    d.axhline(.5,color='#aaaaaa',ls=':',lw=1)
    d.set(title='D  Same concepts at T3: production versus choice',ylabel='French Q (8 ANY templates)',
          xlabel='Successful KO requested templates at T3',xticks=[0,1,2],xticklabels=['0 / 2','1 / 2','2 / 2'],xlim=(-.5,2.5))
    r=s['retained_KO_both_wrappers']
    d.text(.04,.08,f"Among {r['count']} concepts successful on both:\n{r['FR_max_Q8_count']} have FR as their largest Q",transform=d.transAxes,fontsize=9)
    for ax in (a,b,c):ax.set_xticks(range(4),LABELS)
    for ax in axs.flat:
        ax.set_ylim(-.03,1.12);ax.set_yticks([0,.25,.5,.75,1]);ax.grid(axis='y',alpha=.15)
    save(fig,'figure1_choice_production')

    dense=[t for t in s['snapshots'] if t['after_T0_updates']>=0]
    x=np.array([t['after_T0_updates'] for t in dense])
    fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.3,2,1]})
    a,b,c=axs
    fig.suptitle('Saved evaluations every 200 updates: two-template diagnostics',fontsize=15)
    for measure,label,color in [('A','KO requested accuracy','#222222'),('Q_mean','KO choice Q',COLORS['ko'])]:
        vals=np.array([[t['aggregate']['wrappers'][w][measure]['ko'] for w in WRAPPERS] for t in dense])
        a.plot(x,vals.mean(axis=1),label=label,color=color,lw=2)
        a.fill_between(x,vals.min(axis=1),vals.max(axis=1),color=color,alpha=.15)
    a.scatter([0,3000,6000,9000],[eight[p]['Q_mean']['ko'] for p in PHASES],marker='D',s=35,color=COLORS['ko'],edgecolor='white',zorder=4,clip_on=False,label='8-template Q at endpoints')
    a.set(ylabel='Fraction / conditional Q',ylim=(-.03,1.08));a.legend(loc='lower left',ncol=3,fontsize=9)
    matrix=np.array([[np.mean([t['cells'][i][w]['Q']['ko'] for w in WRAPPERS]) for t in dense] for i in ids])
    edges=np.r_[x[0],(x[:-1]+x[1:])/2,x[-1]]
    im=b.pcolormesh(edges,np.arange(61)-.5,matrix,cmap='cividis',vmin=0,vmax=1,rasterized=True)
    b.set(ylabel='Concept index (frozen ID order)',ylim=(59.5,-.5),yticks=[0,14,29,44,59],yticklabels=[1,15,30,45,60])
    fig.colorbar(im,ax=b,label='KO Q: mean of dev1 and dev2',shrink=.85,pad=.02)
    for w,style in zip(WRAPPERS,('-','--')):
        c.plot(x,[t['aggregate']['wrappers'][w]['Z_median'] for t in dense],style,label=w,lw=1.7)
    c.axhline(.9,color='#777777',ls=':',label='Endpoint template criterion: 0.9')
    c.set(ylabel='Median registered mass Z',xlabel='Training updates after T0',ylim=(0,1.08),xticks=[0,1500,3000,4500,6000,7500,9000])
    c.legend(loc='lower right',ncol=3,fontsize=9)
    for ax in axs:
        for lo,hi,color in [(0,3000,COLORS['en']),(3000,6000,COLORS['zh']),(6000,9000,COLORS['fr'])]:
            if ax!=b:ax.axvspan(lo,hi,color=color,alpha=.035)
        for boundary in (3000,6000):ax.axvline(boundary,color='#555555',lw=.8,ls='--')
        ax.set_xlim(0,9000)
    for pos,label in [(1500,'Add EN'),(4500,'Add ZH'),(7500,'Add FR')]:
        a.text(pos,1.04,label,ha='center',fontsize=9)
    save(fig,'figure2_dense_trajectory')
    with (OUT/'concept_dense_trajectories.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['concept_id','phase','phase_step','updates_after_T0','KO_Q_dev1','KO_Q_dev2','Z_dev1','Z_dev2','KO_requested_dev1','KO_requested_dev2'])
        for i in ids:
            for t in dense:
                cells=t['cells'][i]
                writer.writerow([i,t['phase'],t['phase_step'],t['after_T0_updates'],*[cells[w]['Q']['ko'] for w in WRAPPERS],*[cells[w]['Z'] for w in WRAPPERS],*[int(cells[w]['requested']['ko']['success']) for w in WRAPPERS]])

    rows=[]
    for p in PHASES:
        ws=ep[p]['aggregate']['wrappers']
        rows.append(f"| {p} | {ws['dev1']['A_counts']['ko']}/60 · {ws['dev2']['A_counts']['ko']}/60 | {eight[p]['Q_mean']['ko']:.3f} | {eight[p]['Q_mean']['fr']:.3f} | {ws['dev1']['ANY_registered_language_counts']['ko']} · {ws['dev2']['ANY_registered_language_counts']['ko']} | {ws['dev1']['ANY_registered_language_counts']['fr']} · {ws['dev2']['ANY_registered_language_counts']['fr']} |")
    zvals=[t['aggregate']['wrappers'][w]['Z_median'] for t in dense for w in WRAPPERS]
    zep=[z for p in PHASES for z in eight[p]['Z_median_by_wrapper'].values()]
    low=sum(min(t['aggregate']['wrappers'][w]['Z_median'] for w in WRAPPERS)<.9 for t in dense)
    report=f'''# 기존 체크포인트의 능력·선택 궤적

동일한 60개 개념·한국어 뜻풀이 입력에서, 한국어 지정 산출은 T3에도 대부분 성공하지만 ANY 선택의 한국어 점유율은 크게 감소한다. 이는 등록 표현을 요청에 맞춰 산출하는 능력과 ANY 선택의 분리를 지지한다. 일반 한국어 능력의 완전한 보존이나 학습 이력의 독립 인과효과를 검증한 결과는 아니다.

![Figure 1](figure1_choice_production.png)

| 단계 | 한국어 지정 성공 dev1 · dev2 | 평균 Q8(KO) | 평균 Q8(FR) | ANY 한국어 산출 수 dev1 · dev2 | ANY 프랑스어 산출 수 dev1 · dev2 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

T0와 T3에서 모두 두 문구의 한국어 지정 산출에 성공한 개념은 **{r['count']}/60개**다. 그중 **{r['FR_max_Q8_count']}/{r['count']}개**는 T3의 Q8 최대 언어가 프랑스어이고, **{r['KO_Q8_lt_01_count']}/{r['count']}개**는 Q8(KO)<0.1이다. 이 집합의 T3 Q8(KO) 중앙값은 {r['median_T3_Q8_KO']:.4f}다. 최대 확률 언어는 실제 생성 결과와 구분한다.

한국어 지정 정확도도 100%에서 91.7~95.0%로 낮아졌다. 따라서 “능력이 그대로” 또는 ±2%p 동등성 통과라는 표현은 쓰지 않는다. 기존 29/31 통계는 별도 영어 입력의 요청 언어 오류 조건에서 나온 값이므로 이 궤적에 합치지 않았다.

개념별 Q8(KO)>0.5인 수는 T0/T1/T2/T3에서 각각 60/38/8/0이다. T1→T2에 한국어 Q가 증가한 개념도 6개, T2→T3에는 2개 있어 모든 개념의 단조 감소를 주장할 수 없다. 예측 요인 분석이나 외부 검증은 수행하지 않았다.

![Figure 2](figure2_dense_trajectory.png)

저장된 60회 평가의 점수·레코드와 체크포인트 매니페스트 해시를 검증했다. Figure 2에는 T0 종점 이후 46개 시점(200업데이트 간격)을 표시했다. 내부 시점은 원래 두 문구만 측정되어 있으므로 8문구 평균으로 표시하지 않았다. 선은 관측점 사이의 시각적 연결이다. 열지도는 결과로 정렬하지 않은 고정 개념 ID 순서이며 모든 60개 개념을 포함한다.

단계 종점의 8문구별 Z 중앙값 범위는 {min(zep):.3f}~{max(zep):.3f}다. 촘촘한 두 문구 평가의 Z 중앙값은 {min(zvals):.3f}~{max(zvals):.3f}이고, 46개 중 {low}개 시점은 적어도 한 문구가 0.9 미만이다. 특히 새 언어 도입 직후의 하락을 숨기지 않고 하단에 표시했다. 해당 시점의 정규화 Q를 높은 Z가 확인된 종점과 같은 강도로 해석하지 않는다.

**측정 범위.** Q8은 각 문구에서 등록 문자열의 확률로 정규화한 Q를 8문구에 걸쳐 동일 가중 평균한 값이다. 확률을 먼저 합쳐 정규화하지 않았다. Figure 1 A는 두 원래 REQUESTED 문구, B/C/D는 8 ANY 문구이므로 문구 수까지 완전히 같은 비교는 아니다. 두 원래 문구의 ANY 생성 및 Q도 표와 Figure 2에서 별도로 제시했다. T3의 최악 분할 게이트는 여전히 `{eight['T3']['worst_gate']['status']}`이며, Q8을 문구 불변의 개념 특성으로 해석하지 않는다.

**그림 캡션 초안.** Figure 1. Registered-expression production and language choice during sequential language introduction, in one model run and 60 fixed concepts with Korean definition inputs. (A) Requested-language production accuracy; points average two templates and shading spans their values, not a confidence interval. Untrained requested languages are omitted. (B) Conditional language choice averaged over eight ANY templates. (C) Every concept's Korean choice trajectory; blue marks concepts succeeding on both Korean REQUESTED templates at T3. (D) T3 French choice versus the number of successful Korean REQUESTED templates for the same concept. Lines connect stage endpoints and do not estimate within-stage dynamics. These observations do not identify an order effect independent of exposure and recency.

새 학습·새 GPU 평가를 수행하지 않았다. H1 및 본시험은 시작하지 않았다. 하나의 root와 이미 학습한 개념에 대한 기술 통계이며, 언어 추가·최근성·노출량·업데이트 수가 함께 바뀐다. 논문 신규성이나 기존 망각 문헌 대비 우위는 이번 작업에서 검증하지 않았다.

수치와 그림의 단일 출처: `trajectory_summary.json`. 개념별 데이터: `concept_endpoint_trajectories.csv`, `concept_dense_trajectories.csv`. PDF와 SVG도 함께 제공한다.
'''
    (OUT/'report.md').write_text(report)
    print(json.dumps({'output':str(OUT),'dense_min_median_Z':min(zvals),'dense_timepoints_below_09':low,'endpoint_min_median_Z':min(zep)},ensure_ascii=False))


if __name__=='__main__':main()
