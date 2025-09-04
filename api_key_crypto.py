#!/usr/bin/env python3
"""Crypto utilities for encrypting/decrypting an API key with a password.

Public API (minimal main-friendly):
- EncryptedBlob (dataclass): JSON-serializable envelope
- get_api_key(filename: str = "OPENAI.API_KEY") -> str
  Handles create-or-load flow with prompts and secure file writes.

Internals:
- encrypt_api_key(api_key, password) -> EncryptedBlob
- decrypt_api_key(blob, password) -> str
- blob_to_json(blob) -> str
- json_to_blob(text) -> EncryptedBlob

Uses AES-GCM for authenticated encryption and scrypt for password-based key derivation.
Requires: cryptography
"""
from __future__ import annotations

import base64
import getpass
import json
import os
import sys
from dataclasses import dataclass
from typing import Final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend

# ---- Parameters ----
SCRYPT_N: Final[int] = 2 ** 14
SCRYPT_R: Final[int] = 8
SCRYPT_P: Final[int] = 1
KEY_LEN: Final[int] = 32      # 256-bit AES key
SALT_LEN: Final[int] = 16
NONCE_LEN: Final[int] = 12    # AES-GCM standard nonce size
FILENAME_DEFAULT: Final[str] = "OPENAI.API_KEY"

@dataclass
class EncryptedBlob:
    version: int
    salt_b64: str
    nonce_b64: str
    ciphertext_b64: str
    kdf: str = "scrypt"
    n: int = SCRYPT_N
    r: int = SCRYPT_R
    p: int = SCRYPT_P


# --- KDF and AEAD helpers ---

def _derive_key(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    kdf = Scrypt(salt=salt, length=KEY_LEN, n=n, r=r, p=p, backend=default_backend())
    return kdf.derive(password.encode("utf-8"))


def encrypt_api_key(api_key: str, password: str) -> EncryptedBlob:
    import secrets

    salt = secrets.token_bytes(SALT_LEN)
    key = _derive_key(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    nonce = secrets.token_bytes(NONCE_LEN)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, api_key.encode("utf-8"), associated_data=None)
    return EncryptedBlob(
        version=1,
        salt_b64=base64.b64encode(salt).decode("ascii"),
        nonce_b64=base64.b64encode(nonce).decode("ascii"),
        ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
    )


def decrypt_api_key(blob: EncryptedBlob, password: str) -> str:
    if blob.kdf != "scrypt":
        raise ValueError("Unsupported KDF.")
    salt = base64.b64decode(blob.salt_b64)
    nonce = base64.b64decode(blob.nonce_b64)
    ciphertext = base64.b64decode(blob.ciphertext_b64)

    key = _derive_key(password, salt, blob.n, blob.r, blob.p)
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    return plaintext.decode("utf-8")


def blob_to_json(blob: EncryptedBlob) -> str:
    return json.dumps(blob.__dict__, indent=2)


def json_to_blob(text: str) -> EncryptedBlob:
    data = json.loads(text)
    return EncryptedBlob(**data)


# --- File IO helpers and top-level function ---

def _write_secure_file(path: str, content: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        os.chmod(tmp_path, 0o600)
    except Exception:
        pass
    os.replace(tmp_path, path)


def _create_new(filename: str) -> str:
    print("No API key file found. Creating a new encrypted key store.")
    api_key = getpass.getpass("Enter OpenAI API key (input hidden): ").strip()
    if not api_key:
        print("Empty API key; abort.", file=sys.stderr)
        raise SystemExit(1)

    pw1 = getpass.getpass("Create a password to encrypt the key: ")
    pw2 = getpass.getpass("Re-enter the password: ")
    if pw1 != pw2:
        print("Passwords do not match; abort.", file=sys.stderr)
        raise SystemExit(1)

    blob = encrypt_api_key(api_key, pw1)
    _write_secure_file(filename, blob_to_json(blob))
    print(f"Encrypted key saved to ./{filename}")
    return api_key


def _load_existing(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        blob = json_to_blob(text)
    except Exception:
        raise ValueError("Corrupt or unsupported file format.")

    password = getpass.getpass("Enter password to decrypt API key: ")
    try:
        return decrypt_api_key(blob, password)
    except InvalidTag:
        raise ValueError("Decryption failed: incorrect password or file has been modified.")


def get_api_key(filename: str = FILENAME_DEFAULT) -> str:
    """Create-or-load flow. Returns the decrypted API key string.

    Keeps main.py minimal by encapsulating all prompting and IO here.
    """
    if not os.path.exists(filename):
        return _create_new(filename)
    print(f"Existing key store found at ./{filename}")
    try:
        return _load_existing(filename)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)

