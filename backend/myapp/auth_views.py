from django.contrib.auth import logout as django_logout
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .auth_serializers import LoginSerializer, RegisterSerializer
from .models import User


def _user_payload(user: User) -> dict:
    return {
        'id': user.pk,
        'email': user.email,
    }


def _issue_token(user: User) -> str:
    Token.objects.filter(user=user).delete()
    token, _ = Token.objects.get_or_create(user=user)
    return token.key


class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        token_key = _issue_token(user)
        return Response(
            {
                'token': token_key,
                'user': _user_payload(user),
            },
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = LoginSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token_key = _issue_token(user)
        return Response(
            {
                'token': token_key,
                'user': _user_payload(user),
            },
            status=status.HTTP_200_OK,
        )


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        Token.objects.filter(user=request.user).delete()
        django_logout(request)
        return Response({'detail': 'Logged out.'}, status=status.HTTP_200_OK)


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        return Response({'user': _user_payload(request.user)}, status=status.HTTP_200_OK)
