#!/bin/bash
echo "========================================="
echo "🍊 Farminfo GitHub 연결 도우미"
echo "========================================="

# 1. 원격 저장소 재설정
echo "1. 저장소 주소를 설정합니다..."
git remote remove origin 2>/dev/null
git remote add origin https://github.com/fbiyun34-cpu/farminfo.git

# 2. 브랜치 이름 강제 설정
git branch -M main

# 3. 푸시 시도
echo "2. GitHub에 코드를 업로드합니다."
echo "   ⚠️  로그인 창이 뜨거나, 아이디/비밀번호를 물어볼 수 있습니다."
echo "   ⚠️  비밀번호란에는 GitHub Password 대신 'Personal Access Token'을 입력해야 할 수도 있습니다."

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 성공적으로 연결되었습니다!"
    echo "이제 Streamlit Cloud에서 배포를 진행해주세요."
else
    echo ""
    echo "❌ 연결에 실패했습니다. 다시 시도하거나 인터넷 연결을 확인해주세요."
fi
