# Tutorial
# This tutorial walks you through some of the fundamental Airflow concepts, objects,
# and their usage while writing your first pipeline.


# Example pipeline definition
# Here is an example of a basic pipeline definition.
# Do not worry if this looks complicated, a line by line explanation follows below.
# from _sqlite3 import
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
# from airflow.models import O
default_args = {
    'owner': 'will',
    'depends+on_past': False,
    'start_date': datetime(2018, 12, 10),
    'email': ['joonable2@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG('airflow_tutorial', default_args=default_args, schedule_interval=timedelta(days=1))

t1 = BashOperator(
    task_id='print_date',
    bash_command='date',
    dag=dag
)

t2 = BashOperator(
    task_id='sleep',
    bash_command='sleep 5',
    retries=3,
    dag=dag
)

templated_command = """
    {% for i in range(5) %}
        echo "{{ ds }}"
        echo "{{ macros.ds_add(ds, 7) }}"
        echo "{{ params.my_param }}"
    {% endfor %}
"""

t3 = BashOperator(
    task_id='templated',
    bash_command=templated_command,
    params={'my_param': 'Parameter I passed in'},
    dag=dag
)

t2.set_upstream(t1)
t3.set_upstream(t1)


# t1.set_downstream(t2)
#
# # This means that t2 will depend on t1
# # running successfully to run.
# # It is equivalent to:
# t2.set_upstream(t1)
#
# # The bit shift operator can also be
# # used to chain operations:
# t1 >> t2
#
# # And the upstream dependency with the
# # bit shift operator:
# t2 << t1
#
# # Chaining multiple dependencies becomes
# # concise with the bit shift operator:
# t1 >> t2 >> t3
#
# # A list of tasks can also be set as
# # dependencies. These operations
# # all have the same effect:
# t1.set_downstream([t2, t3])
# t1 >> [t2, t3]
# [t2, t3] << t1

# It’s a DAG definition file
# One thing to wrap your head around (it may not be very intuitive for everyone at first) is
# that this Airflow Python script is really just a configuration file specifying the DAG’s structure as code.
#
# The actual tasks defined here will run in a different context from the context of this script.
# Different tasks run on different workers at different points in time,
# which means that this script cannot be used to cross communicate between tasks.
# Note that for this purpose we have a more advanced feature called XCom.
#
# People sometimes think of the DAG definition file as a place where they can do some actual data processing
# - that is not the case at all! The script’s purpose is to define a DAG object.
# It needs to evaluate quickly (seconds, not minutes)
# since the scheduler will execute it periodically to reflect the changes if any.