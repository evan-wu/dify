"""merge heads

Revision ID: b90b587766cb
Revises: cf8f4fc45278, 3b6551091d65
Create Date: 2024-12-17 08:52:00.510494

"""
from alembic import op
import models as models
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b90b587766cb'
down_revision = ('cf8f4fc45278', '3b6551091d65')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
