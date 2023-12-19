#！ /bin/bash 
#clean up some interm-data

# set -x

if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

folder_path="$1"
models_folder="${folder_path}models"
echo "$models_folder"


if [ ! -d "$models_folder" ]; then
    echo "Error: $models_folder does not exits."
    exit 1
fi

for file in "$models_folder"/*.p; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        
        if [[ $filename =~ [0-9]+ ]]; then
            number=`echo $filename | tr -cd "[0-9]" `
            number=`echo $number |sed 's/^0\+//'`

            if [[ $number%10 -ne 0 ]]; then
                echo $number
                echo "Deleting $file"
                rm "$file"
            fi
        fi
    fi
done

for agent_folder in "${models_folder}"/agent_{0,1}*; do
    if [ -d "$agent_folder" ]; then
        echo "Processing $agent_folder ..."

        for file in "$agent_folder"/*.p; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                
                if [[ $filename =~ [0-9]+ ]]; then
                    number=`echo $filename | tr -cd "[0-9]" `
                    number=`echo $number |sed 's/^0\+//'`

                    if [[ $number%10 -ne 0 ]]; then
                        echo $number
                        echo "Deleting $file"
                        rm "$file"
                    fi
                fi
            fi
        done
    fi
done

echo "Clean up completed!"