import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def frequency_encode(df: pd.DataFrame, col: str, new_col: str = None) -> pd.DataFrame:
    """Replace values in `col` with their occurrence count in `df`."""
    new_col = new_col or f'{col}_freq'
    df[new_col] = df[col].map(df[col].value_counts())
    return df


def plot_columns(df: pd.DataFrame, cols: list[str], target: str = 'isFraud', categorical_max_unique: int = 20, ncols: int = 3):
    """Grid-plot a list of columns, auto-picking chart type per column.

    - Categorical (dtype object/category, or numeric with <= categorical_max_unique
      unique values): sns.barplot of fraud rate per category, with n= labels.
    - Numerical (everything else): sns.boxplot split by target (log-scale y if
      the column spans several orders of magnitude), for a quick fraud-vs-not view.
    """
    n = len(cols)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, cols):
        is_categorical = (
            df[col].dtype == object
            or df[col].dtype.name == 'category'
            or df[col].nunique(dropna=True) <= categorical_max_unique
        )

        if is_categorical:
            stats = df.groupby(col)[target].agg(['mean', 'count']).sort_values('mean', ascending=False)
            sns.barplot(x=stats.index, y=stats['mean'], color='salmon', edgecolor='gray', ax=ax)
            ax.set_ylabel('Fraud Rate')
            ax.set_xlabel('')
            for i, (rate, cnt) in enumerate(zip(stats['mean'], stats['count'])):
                ax.text(i, rate, f'n={cnt:,}', ha='center', va='bottom', fontsize=7)
        else:
            plot_df = df[[col, target]].dropna().copy()
            plot_df[target] = plot_df[target].map({0: 'Not Fraud', 1: 'Fraud'})
            sns.boxplot(data=plot_df, x=target, y=col, hue=target,
                        palette={'Not Fraud': 'lightblue', 'Fraud': 'salmon'},
                        legend=False, ax=ax)
            ax.set_xlabel('')
            if (plot_df[col] > 0).all() and plot_df[col].max() / plot_df[col].min() > 100:
                ax.set_yscale('log')

        ax.set_title(col, fontsize=10)
        ax.tick_params(axis='x', labelsize=8, rotation=45)

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def ploting_cnt_amt(df, col, lim=2000):
    df = df.copy()
    df[col] = df[col].astype(object).fillna('NaN')

    total=len(df)
    total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    
    plt.figure(figsize=(16,14))    
    plt.suptitle(f'{col} Distributions ', fontsize=24)
    
    plt.subplot(211)
    g = sns.countplot( x=col,  data=df, order=list(tmp[col].values),
                       hue=col, palette='viridis', legend=False)
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                       color='crimson', legend=False, )
    gt.set_ylim(0,tmp['Fraud'].max()*1.1)
    gt.set_ylabel("%Fraud Transactions", fontsize=16)
    g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
    g.set_xlabel(f"{col} Category Names", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    sizes = []
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=12) 
        
    g.set_ylim(0,max(sizes)*1.15)
    
    #########################################################################
    perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum() \
                / df.groupby([col])['TransactionAmt'].sum() * 100).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    amt = df.groupby([col])['TransactionAmt'].sum().reset_index()
    perc_amt = perc_amt.fillna(0)
    plt.subplot(212)
    g1 = sns.barplot(x=col, y='TransactionAmt',
                       data=amt,
                       order=list(tmp[col].values),
                       hue=col, palette='magma', legend=False)
    g1t = g1.twinx()
    g1t = sns.pointplot(x=col, y='Fraud', data=perc_amt,
                        order=list(tmp[col].values),
                       color='crimson', legend=False, )
    g1t.set_ylim(0,perc_amt['Fraud'].max()*1.1)
    g1t.set_ylabel("%Fraud Total Amount", fontsize=16)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    g1.set_title(f"{col} by Transactions Total + %of total and %Fraud Transactions", fontsize=20)
    g1.set_xlabel(f"{col} Category Names", fontsize=16)
    g1.set_ylabel("Transaction Total Amount(U$)", fontsize=16)
    g1.set_xticklabels(g.get_xticklabels(),rotation=45)    
    
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt*100),
                ha="center",fontsize=12) 
        
    plt.subplots_adjust(hspace=.4, top = 0.9)
    plt.show()

def ploting_dist_ratio(df, col, lim=2000):
    df = df.copy()
    df[col] = df[col].astype(object).fillna('NaN')
    total=len(df)
    total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=(20,5))
    plt.suptitle(f'{col} Distributions ', fontsize=22)

    plt.subplot(121)
    g = sns.countplot(x=col, data=df, order=list(tmp[col].values))
    # plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])
    g.set_title(f"{col} Distribution\nCound and %Fraud by each category", fontsize=18)
    g.set_ylim(0,400000)
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    gt.set_ylim(0,20)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)
    g.set_xlabel(f"{col} Category Names", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    for p in gt.patches:
        height = p.get_height()
        gt.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=14) 
        
    perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum() / total_amt * 100).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.subplot(122)
    g1 = sns.boxplot(x=col, y='TransactionAmt', hue='isFraud', 
                     data=df[df['TransactionAmt'] <= lim], order=list(tmp[col].values))
    g1t = g1.twinx()
    g1t = sns.pointplot(x=col, y='Fraud', data=perc_amt, order=list(tmp[col].values),
                       color='black', legend=False, )
    g1t.set_ylim(0,5)
    g1t.set_ylabel("%Fraud Total Amount", fontsize=16)
    g1.set_title(f"{col} by Transactions dist", fontsize=18)
    g1.set_xlabel(f"{col} Category Names", fontsize=16)
    g1.set_ylabel("Transaction Amount(U$)", fontsize=16)
        
    plt.subplots_adjust(hspace=.4, wspace = 0.35, top = 0.80)
    
    plt.show()