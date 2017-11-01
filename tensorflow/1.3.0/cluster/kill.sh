ps -ef | grep -v grep | grep _cl | awk '{print $2}' | xargs -i kill -9 {}
