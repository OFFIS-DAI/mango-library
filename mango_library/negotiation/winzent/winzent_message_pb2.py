# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: winzent_message.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='winzent_message.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\x15winzent_message.proto\"\xa5\x01\n\x0eWinzentMessage\x12\x10\n\x08msg_type\x18\x01 \x01(\x05\x12\x0b\n\x03ttl\x18\x02 \x01(\x05\x12\x11\n\ttime_span\x18\x03 \x03(\x03\x12\n\n\x02id\x18\x04 \x01(\t\x12\x0e\n\x06sender\x18\x05 \x01(\t\x12\x11\n\tis_answer\x18\x06 \x01(\x08\x12\r\n\x05value\x18\x07 \x03(\x05\x12\x11\n\tanswer_to\x18\x08 \x01(\t\x12\x10\n\x08receiver\x18\t \x01(\tb\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_WINZENTMESSAGE = _descriptor.Descriptor(
  name='WinzentMessage',
  full_name='WinzentMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_type', full_name='WinzentMessage.msg_type', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ttl', full_name='WinzentMessage.ttl', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='time_span', full_name='WinzentMessage.time_span', index=2,
      number=3, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='id', full_name='WinzentMessage.id', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sender', full_name='WinzentMessage.sender', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='is_answer', full_name='WinzentMessage.is_answer', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='WinzentMessage.value', index=6,
      number=7, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='answer_to', full_name='WinzentMessage.answer_to', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='receiver', full_name='WinzentMessage.receiver', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=26,
  serialized_end=191,
)

DESCRIPTOR.message_types_by_name['WinzentMessage'] = _WINZENTMESSAGE

WinzentMessage = _reflection.GeneratedProtocolMessageType('WinzentMessage', (_message.Message,), dict(
  DESCRIPTOR = _WINZENTMESSAGE,
  __module__ = 'winzent_message_pb2'
  # @@protoc_insertion_point(class_scope:WinzentMessage)
  ))
_sym_db.RegisterMessage(WinzentMessage)


# @@protoc_insertion_point(module_scope)
