def summarize_stats(df):
    print("YOU CAN'T HANDLE THE STATS!!!!!!")
    print('|------------------------------------------------------|')
    print('|------------------------------------------------------|')
    print(f'Shape: {df.shape}')
    print('|------------------------------------------------------|')
    print(df.info())
    print('|------------------------------------------------------|')
    print('|------------------------------------------------------|')
    for col in df.columns:
        print(f'|-------{col}-------|')
        print()
        print(f'dtpye: {df[col].dtype}')
        print()
        print(f'Null count: {df[col].isnull().sum()}')
        print()
        print(df[col].describe())
        print()
        print(df[col].value_counts())
        print()
        print(df[col].unique())
        print()
        if df[col].dtype != 'O':
            sns.distplot(df[col])
            plt.title(f'Distribution of {col}')
            plt.ylabel('Frequency')
            plt.xlabel(col)
            plt.show()
        print('|------------------------------------------------------|')
        print('|------------------------------------------------------|')



def viz_outliers(df, k):
    for col in df.columns:
        if df[col].dtype != 'O':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper_bound =  q3 + k * iqr
            lower_bound =  q1 - k * iqr
            ax=sns.distplot(df[col])
            plt.axvline(lower_bound)
            plt.axvline(upper_bound)
            plt.title(f'Distribution of {col} with \n upper and lower bounds')
            plt.show()



def remove_outliers(df, cols, k):
    for col in df[cols]:
        q1, q3 = df[col].quantile([0.25, 0.75)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df