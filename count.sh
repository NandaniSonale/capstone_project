pacman -S mingw-w64-x86_64-ffmpeg --noconfirm
ffprobe -v error -select_streams v:0 -show_entries frame=pict_type -of default=noprint_wrappers=1:nokey=1 "/c/Users/newuser/capstone_project/Human Activity Recognition - Video Dataset/Walking/Walking (1).mp4" > types.txt
echo "I-Frames:"
grep -c I types.txt
echo "P-Frames:"
grep -c P types.txt
echo "B-Frames:"
grep -c B types.txt
