import pandas as pd
from datetime import datetime
import os

# Get the current working directory (optional for clarity)
current_dir = os.getcwd()

# Construct the full path to the file (assuming it's in the same directory)
file_path = os.path.join(current_dir, "HI-Small_Trans.csv")

# Read data as a pandas DataFrame
raw = pd.read_csv(file_path, dtype=str)

# Initialize dictionaries for tracking unique values
currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

# Define header row
header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,Is Laundering\n"

# Initialize variables
firstTs = -1

# Open output file for writing
with open(os.path.splitext(file_path)[0] + "_formatted.csv", 'w') as writer:
    writer.write(header)
    for i in range(len(raw)):
        # Convert timestamp string to datetime object
        datetime_object = datetime.strptime(raw.loc[i, "Timestamp"], '%Y/%m/%d %H:%M')
        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        hour = datetime_object.hour
        minute = datetime_object.minute

        # Calculate relative timestamp if first timestamp encountered
        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        ts = ts - firstTs

        # Get dictionary values for currencies and payment format
        cur1 = get_dict_val(raw.loc[i, "Receiving Currency"], currency)
        cur2 = get_dict_val(raw.loc[i, "Payment Currency"], currency)
        fmt = get_dict_val(raw.loc[i, "Payment Format"], paymentFormat)

        # Combine account information and get unique identifier
        fromAccIdStr = raw.loc[i, "From Bank"] + raw.loc[i, "Account"]
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = raw.loc[i, "To Bank"] + raw.loc[i, "Account.1"]
        toId = get_dict_val(toAccIdStr, account)

        # Convert amount strings to floats
        amountReceivedOrig = float(raw.loc[i, "Amount Received"])
        amountPaidOrig = float(raw.loc[i, "Amount Paid"])

        # Extract laundering flag
        isl = int(raw.loc[i, "Is Laundering"])

        # Format output line
        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                    (i,fromId,toId,ts,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,isl)

        # Write line to output file
        writer.write(line)