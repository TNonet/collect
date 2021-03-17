import pyspark
import datetime
import numpy as np
from pyspark.sql import Window
import pyspark.sql.functions as func
from pyspark.sql.types import (StringType, FloatType, DoubleType, ByteType, ShortType,
                               IntegerType, LongType, TimestampType, DateType)

from typing import List, Callable, Optional, Union, Any, Dict, Iterable


def collect(df: pyspark.sql.dataframe.DataFrame,
            groupBy: Union[str, List[str]],
            orderBy: Union[str, List[str]],
            columns_to_collect: List[str],
            columns_to_keep_min_value: Optional[Union[str, Iterable[str]]] = (),
            alias_format: Callable[[str], str] = lambda x: x + '_list',
            count: bool = True,
            fillna: bool = True):
    """Collects (Aggregates into a list) `columns_to_collect`, columns of df, into a list
    that is ordered by `orderBy` and grouped by `groupBy`.

    Parameters
    ----------
    df : pyspark.sql.dataframe.DataFrame
        dataframe that will be ordered and grouped and then `columns_to_collect` will be collected
    groupBy: str or list of strs
        columns that will group df
    orderBy: str or list of strs
        columns that will order df
    columns_to_collect : list of str
        Columns of `df` to collect into a list
    columns_to_keep_min_value : list of str, optional
        Columns of `df` to keep the first value of each group
        if columns_to_keep_min_value is not set, 'day', 'month', and 'year' will be kept
    alias_format : Callable with str to return str
        Method for renaming/aliasing collected columns
    count : bool
        Flag whether to count the number of rows in each row
    fillna : bool
        Flag whether to fill `NULL` values with non-`NULL` but obvious data as a placeholder
        If `fillna` == False:
            columns with `NULL` values will have a collected list that is short as `NULL`
            values are dropped when collected.

    Returns
    -------
    aggregated_group : pyspark.sql.dataframe.DataFrame
        Aggregated dataframe

    Examples
    --------
    >>>df_collected = collect(df, groupBy=['A', 'B'],
    ...                           orderBy='Time',
    ...                           columns_to_collect=['X', 'Y'],
    ...                           columns_to_keep_min_value=['day', 'month', 'year'])
    ...df_collected
    DataFrame[A: string, B: string, X_list: array<float>, 
              Y_list: array<smallint>, day: string, month: string, year: string, n: bigint]
    """
    if fillna:
        df = replace_nulls(df)

    w = Window.partitionBy(groupBy).orderBy(orderBy)
    for col_name in columns_to_collect:
        df = df.withColumn(alias_format(col_name), func.collect_list(col_name).over(w))

    grouped_df = df.groupBy(groupBy)

    agg_list = [func.max(alias_format(col_name)).alias(alias_format(col_name)) for col_name in columns_to_collect]

    for col_name in columns_to_keep_min_value:
        agg_list.append(func.min(func.col(col_name)).alias(col_name))

    if count:
        agg_list.append(func.count(alias_format(columns_to_collect[0])).alias('n'))
    return grouped_df.agg(*agg_list)


def get_nan_dict(spark=True) -> Dict[str, Any]:
    """ Returns a mapping of SparkTypes name to their respective python NULL equivalents.
    When `pyspark.sql.functions.collect_list` is used, NULL values are dropped. To prevent this,
    we replace the NULL values with the values from get_nan_dict(spark=True).

    The corresponding python equivalents are found from get_nan_dict(spark=False)

    Parameters
    ----------
    spark : bool
        Flag wheter values should be spark acceptable or numpy/pandas acceptable.

    Returns
    -------
    nan_dict : dict
        Mapping of the name of a SparkType to its replacement value


    Notes
    -----
    +-------------------------------------------------------------+
    |               `NULL` Replacement Chart                      |
    +----------------+-------------------+------------------------+
    | PySpark Type   |    Python Type    | 'NUll' Value           |
    |----------------+-------------------+------------------------+
    | StringType     | str               | 'NULL'                 |
    | FloatType      | float             | float('nan')           |
    | DoubleType     | float             | float('nan')           |
    | ByteType       | int               | -128                   |
    | ShortType      | int               | -32768                 |
    | IntegerType    | int               | -2**31                 |
    | LongType       | int               | -2*63                  |
    | TimestampType  | datetime.datetime | fromtimestamp(0.)      |
    | DateType       | datetime.date     | str(fromtimestamp(0.)) |
    +----------------+-------------------+------------------------+
    """
    base_time = 0.
    if spark:
        return {StringType: 'NULL', FloatType: float('nan'), DoubleType: float('nan'), ByteType: -int(2 ** 7),
                ShortType: -int(2 ** 15), IntegerType: -int(2 ** 31), LongType: -int(2 ** 63), TimestampType: base_time,
                DateType: str(datetime.date.fromtimestamp(base_time))}
    else:
        nan_dict = get_nan_dict(spark=True)
        nan_dict[FloatType] = np.float32(nan_dict[FloatType])
        nan_dict[DoubleType] = np.float64(nan_dict[FloatType])
        nan_dict[ByteType] = np.int8(nan_dict[ByteType])
        nan_dict[ShortType] = np.int16(nan_dict[ShortType])
        nan_dict[IntegerType] = np.int32(nan_dict[IntegerType])
        nan_dict[LongType] = np.int64(nan_dict[LongType])
        nan_dict[TimestampType] = np.datetime64(datetime.date.fromtimestamp(base_time))
        nan_dict[DateType] = datetime.date.fromtimestamp(base_time)
        return nan_dict


def replace_nulls(df: pyspark.sql.dataframe.DataFrame):
    """ Since pyspark.sql.functions.collect_list will drop `NULL` values, we must replace nulls with valid
    but non-sensical values so later users can see that the value was "NULL"

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
        dataframe in which `NULL` values are to be replaced

    Returns
    -------
    no_nulls : pyspark.sql.dataframe.DataFrame
        `df` with `NULL` values replaced according to '`NULL` Replacement Chart' in Notes Section

    Examples
    --------
    >>>df = spark.read.table("table_with_nulls")
    ...df_no_nulls = replace_nulls(df)

    df_no_nulls will have no nulls
    """
    nan_dict = get_nan_dict(spark=True)

    simple_string = {k().simpleString(): k for k in nan_dict.keys()}
    # .simpleString() is how a type is formatted in df.dtypes

    fill_dict = {column: nan_dict[simple_string[dtype]] for column, dtype in df.dtypes}

    return df.fillna(fill_dict)
